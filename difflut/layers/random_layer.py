import torch
import torch.nn as nn
from typing import Type, Dict, Any, Optional
from .base_layer import BaseLUTLayer
from ..registry import register_layer

@register_layer("random")
class RandomLayer(BaseLUTLayer):
    """
    LUT layer with purely random fixed mapping.
    Each input is used at least once per node before being reused.
    Connections are randomly initialized and remain fixed during training.
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int, 
                 node_type: Type[nn.Module],
                 node_kwargs: Optional[Dict[str, Any]] = None,
                 seed: int = 42):
        """
        Args:
            input_size: Size of input vector (from encoder or previous layer)
                       Should match: (batch_size, input_size)
            output_size: Number of LUT nodes (output will be batch_size, output_size * output_dim)
            node_type: LUT node class to use
            node_kwargs: Additional node arguments (should include input_dim and output_dim)
                        Dimension spec: nodes expect (batch_size, output_size, node_input_dim)
            seed: Random seed for reproducible mapping
        """
        self.seed = seed
        
        # Initialize parent (n will be extracted from created nodes)
        super().__init__(input_size, output_size, node_type, node_kwargs)
        
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
        
        # Flatten mapping to (output_size * n,) for single index_select operation
        mapping_flat = self._mapping.reshape(-1)  # (output_size * n,)
        
        # Index select along input dimension: x -> (batch_size, output_size * n)
        # x: (batch_size, input_size)
        # Select the indices from mapping_flat along dimension 1
        mapped_flat = torch.index_select(x, 1, mapping_flat)  # (batch_size, output_size * n)
        
        # Reshape to (batch_size, output_size, n)
        mapped_inputs = mapped_flat.reshape(batch_size, self.output_size, self.n)
        
        return mapped_inputs
    
    def get_mapping_matrix(self) -> torch.Tensor:
        """Get the random mapping matrix for inspection."""
        return self._mapping.clone()
    
    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return f"input_size={self.input_size}, output_size={self.output_size}, " \
               f"n={self.n}, seed={self.seed}, mapping=random"