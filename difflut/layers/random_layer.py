import torch
import torch.nn as nn
from typing import Type
from .base_layer import BaseLUTLayer
from ..registry import register_layer
from ..nodes.node_config import NodeKwargs


@register_layer("random")
class RandomLayer(BaseLUTLayer):
    """
    LUT layer with purely random fixed mapping.
    Each input is used at least once per node before being reused.
    Connections are randomly initialized and remain fixed during training.
    
    Uses efficient index_select operations to minimize memory during mapping,
    avoiding large intermediate tensors.
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int, 
                 node_type: Type[nn.Module],
                 node_kwargs: NodeKwargs = None,
                 seed: int = 42,
                 flip_probability: float = 0.0,
                 grad_stabilization: str = 'none',
                 grad_target_std: float = 1.0,
                 grad_subtract_mean: bool = False,
                 grad_epsilon: float = 1e-8):
        """
        Args:
            input_size: Size of input vector (from encoder or previous layer)
                       Should match: (batch_size, input_size)
            output_size: Number of LUT nodes (output will be batch_size, output_size * output_dim)
            node_type: LUT node class to use
            node_kwargs: Node configuration (NodeConfig instance or dict with input_dim, output_dim, etc.)
                        Dimension spec: nodes expect (batch_size, output_size, node_input_dim)
            seed: Random seed for reproducible mapping
            flip_probability: Probability of flipping each bit during training (0.0 to 1.0)
            grad_stabilization: Gradient stabilization mode ('none', 'layerwise', 'batchwise')
            grad_target_std: Target standard deviation for gradient rescaling
            grad_subtract_mean: Whether to subtract mean before rescaling
            grad_epsilon: Small constant for numerical stability
        """
        self.seed = seed
        
        # Initialize parent (n will be extracted from created nodes)
        super().__init__(input_size, output_size, node_type, node_kwargs, flip_probability,
                        grad_stabilization, grad_target_std, grad_subtract_mean, grad_epsilon)
        
        # Initialize the random mapping
        self._init_mapping()
    
    def _init_mapping(self):
        """
        Initialize random mapping matrix.
        Ensures each input is used at least once per node before any reuse.
        """
        # Store current RNG state and set seed for reproducibility
        rng_state = torch.get_rng_state()
        torch.manual_seed(self.seed)
        
        mapping = torch.empty((self.output_size, self.n), dtype=torch.long)

        for node_idx in range(self.output_size):
            # Calculate how many full cycles we need
            full_cycles = self.n // self.input_size
            remainder = self.n % self.input_size
            
            indices = []
            
            # For each full cycle, use all inputs once in random order
            for _ in range(full_cycles):
                perm = torch.randperm(self.input_size)
                indices.extend(perm.tolist())
            
            # For remainder, use a random subset of inputs
            if remainder > 0:
                perm = torch.randperm(self.input_size)
                indices.extend(perm[:remainder].tolist())
            
            # Store mapping for this node
            mapping[node_idx, :] = torch.tensor(indices, dtype=torch.long)
        
        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('_mapping', mapping)
        
        # OPTIMIZATION: Pre-compute flattened mapping for faster indexing
        # This avoids repeated reshape operations during forward pass
        self.register_buffer('_mapping_flat', mapping.reshape(-1))
        
        # Restore original RNG state
        torch.set_rng_state(rng_state)

    def get_mapping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get mapped inputs using the fixed random mapping.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Mapped inputs of shape (batch_size, output_size, n)
        """
        batch_size = x.shape[0]
        
        # MEMORY OPTIMIZATION: Avoid expanding (batch_size, output_size, input_size) tensor
        # which would be massive for large batches (e.g., batch=73728, output_size=3578, n=4704 = 8.6 GB)
        # 
        # Shape of self._mapping: (output_size, n) containing indices into input dimension
        # We want: output[b, o, i] = x[b, mapping[o, i]]
        #
        # The batch dimension is independent of the mapping - each sample uses the same mapping.
        # Use torch.index_select to efficiently gather along the input dimension without 
        # expanding intermediate tensors.
        
        # OPTIMIZATION: Use pre-computed flattened mapping to avoid reshape during forward pass
        # Index select along input dimension: x -> (batch_size, output_size * n)
        # x: (batch_size, input_size)
        # Select the indices from pre-computed _mapping_flat along dimension 1
        mapped_flat = torch.index_select(x, 1, self._mapping_flat)  # (batch_size, output_size * n)
        
        # Reshape to (batch_size, output_size, n)
        mapped_inputs = mapped_flat.reshape(batch_size, self.output_size, self.n)
        
        return mapped_inputs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through random mapping and node.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output tensor of shape (batch_size, output_size * output_dim)
        """
        # Validate input dimensions
        self._validate_input_dims(x)
        
        # Standard approach: get mapping then forward through node
        # PyTorch handles gradients automatically
        mapped_inputs = self.get_mapping(x)
        output = self.node(mapped_inputs)
        
        # Output shape: (batch_size, output_size, output_dim)
        # Reshape to 2D for next layer: (batch_size, output_size * output_dim)
        batch_size = output.shape[0]
        output = output.view(batch_size, -1)
        
        return output
    
    def get_mapping_matrix(self) -> torch.Tensor:
        """Get the random mapping matrix for inspection."""
        return self._mapping.clone()
    
    def extra_repr(self) -> str:
        """String representation for print(model)."""
        flip_str = f", flip_prob={self.flip_probability}" if self.flip_probability > 0 else ""
        grad_str = f", grad_stab={self.grad_stabilization}" if self.grad_stabilization != 'none' else ""
        return f"input_size={self.input_size}, output_size={self.output_size}, " \
               f"n={self.n}, seed={self.seed}, mapping=random{flip_str}{grad_str}"