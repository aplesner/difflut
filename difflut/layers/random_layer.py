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
        
        Creates an index tensor of shape (output_size, n) where each entry specifies
        which input index to use. This is more memory efficient than the binary mask.
        """
        # Store current RNG state and set seed for reproducibility
        rng_state = torch.get_rng_state()
        torch.manual_seed(self.seed)
        
        # Create index mapping: (output_size, n)
        # For each output node and position, store which input index to use
        # Use int16 for memory efficiency (supports up to 32k input features)
        dtype = torch.int16 if self.input_size < 32768 else torch.int32
        mapping_indices = torch.zeros((self.output_size, self.n), dtype=dtype)

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
            
            # Store indices for this node
            mapping_indices[node_idx] = torch.tensor(indices, dtype=dtype)
        
        # Register as buffer (not a parameter, but saved with model)
        # Shape: (output_size, n) - index mapping (int16/int32)
        # Memory: output_size * n * 2 bytes (vs input_size * output_size * n * 1 byte for mask)
        # For typical case: 1000 * 6 * 2 = 12KB (vs 1568 * 1000 * 6 * 1 = 9.4MB)
        # Reduction: ~99% memory savings for mapping storage!
        self.register_buffer('_mapping_indices', mapping_indices)
        
        # Restore original RNG state
        torch.set_rng_state(rng_state)

    def get_mapping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get mapped inputs using the fixed random mapping.
        
        Memory-optimized version using advanced indexing with cached indices.
        Avoids creating intermediate float tensors and directly gathers values.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Mapped inputs of shape (batch_size, output_size, n)
        """
        # MEMORY OPTIMIZATION: Use gather with index tensor
        # This is more memory efficient than einsum as it:
        # 1. Doesn't require float conversion of mask
        # 2. Uses optimized indexing kernels
        # 3. Avoids intermediate sparse matmul operations
        
        batch_size = x.shape[0]
        
        # Expand indices for batch dimension: (1, output_size, n) -> (batch_size, output_size, n)
        indices_expanded = self._mapping_indices.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Convert to long for indexing
        indices_long = indices_expanded.long()
        
        # Gather values: for each (batch, output, pos), get x[batch, indices[output, pos]]
        # x: (batch_size, input_size)
        # We need to gather from input_size dimension using indices
        # Expand x to match output dimension, then gather along input dimension
        
        # Approach: Use batched index_select equivalent via gather
        # x.unsqueeze(1): (batch_size, 1, input_size)
        # expand to: (batch_size, output_size, input_size)
        # then gather along dim=2 using indices: (batch_size, output_size, n)
        x_expanded = x.unsqueeze(1).expand(-1, self.output_size, -1)
        mapped_inputs = torch.gather(x_expanded, dim=2, index=indices_long)
        
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
        """Get the random mapping matrix for inspection (as indices)."""
        # Return the index mapping directly
        return self._mapping_indices.long()
    
    def extra_repr(self) -> str:
        """String representation for print(model)."""
        flip_str = f", flip_prob={self.flip_probability}" if self.flip_probability > 0 else ""
        grad_str = f", grad_stab={self.grad_stabilization}" if self.grad_stabilization != 'none' else ""
        return f"input_size={self.input_size}, output_size={self.output_size}, " \
               f"n={self.n}, seed={self.seed}, mapping=random{flip_str}{grad_str}"