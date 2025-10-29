import torch
import torch.nn as nn
from typing import Type, Dict, Any, Optional
from .base_layer import BaseLUTLayer
from ..registry import register_layer


class RandomMappingFunction(torch.autograd.Function):
    """
    Custom autograd function that fuses random mapping with node forward pass.
    This avoids storing the intermediate mapped_inputs tensor for backward pass,
    significantly reducing memory usage during backpropagation.
    """
    
    @staticmethod
    def forward(ctx, x, mapping, node, output_size, n):
        """
        Forward pass: map inputs and pass to node in one operation.
        
        Args:
            x: Input tensor (batch_size, input_size)
            mapping: Mapping buffer (output_size, n) with indices into input_size
            node: The LUT node module
            output_size: Number of nodes
            n: Node input dimension
        
        Returns:
            output: Result of node forward pass (batch_size, output_size, output_dim)
        """
        batch_size = x.shape[0]
        
        # Perform mapping: (batch_size, input_size) -> (batch_size, output_size, n)
        mapping_flat = mapping.reshape(-1)
        mapped_flat = torch.index_select(x, 1, mapping_flat)
        mapped_inputs = mapped_flat.reshape(batch_size, output_size, n)
        
        # Forward through node
        output = node(mapped_inputs)
        
        # Save for backward: save references to inputs and mapping info
        # We save the mapping indices and x, but NOT the intermediate mapped_inputs
        ctx.save_for_backward(x, mapping)
        ctx.node = node
        ctx.output_size = output_size
        ctx.n = n
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: recompute mapping during backprop to avoid storing intermediate.
        
        Args:
            grad_output: Gradient w.r.t. output (batch_size, output_size, output_dim)
        
        Returns:
            Gradients for: x, mapping, node, output_size, n
        """
        x, mapping = ctx.saved_tensors
        node = ctx.node
        output_size = ctx.output_size
        n = ctx.n
        batch_size = x.shape[0]
        
        # Recompute mapping in backward pass
        mapping_flat = mapping.reshape(-1)
        mapped_flat = torch.index_select(x, 1, mapping_flat)
        mapped_inputs = mapped_flat.reshape(batch_size, output_size, n)
        
        # Set up for backward through node
        mapped_inputs.requires_grad = True
        
        # Recompute node forward to get intermediate values for backward
        with torch.enable_grad():
            output = node(mapped_inputs)
        
        # Backward through node
        torch.autograd.backward(output, grad_output)
        
        # Get gradient w.r.t. mapped_inputs from the node
        grad_mapped_inputs = mapped_inputs.grad
        
        # Backward through mapping: (batch_size, output_size, n) -> (batch_size, input_size)
        # We need to scatter gradients back to the original input positions
        grad_x = torch.zeros_like(x)
        
        batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1, 1)
        batch_indices = batch_indices.expand(-1, output_size, n)
        
        mapping_expanded = mapping.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Scatter_add gradients back to input positions
        grad_x.scatter_add_(1, mapping_expanded.reshape(batch_size, -1), 
                            grad_mapped_inputs.reshape(batch_size, -1))
        
        # No gradients for other inputs
        return grad_x, None, None, None, None

@register_layer("random")
class RandomLayer(BaseLUTLayer):
    """
    LUT layer with purely random fixed mapping.
    Each input is used at least once per node before being reused.
    Connections are randomly initialized and remain fixed during training.
    
    MEMORY OPTIMIZATION: Uses fused mapping+forward operation to avoid storing
    intermediate activations (batch_size, output_size, n) during backward pass.
    For large batch sizes, this can reduce memory by ~50-70%.
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int, 
                 node_type: Type[nn.Module],
                 node_kwargs: Optional[Dict[str, Any]] = None,
                 seed: int = 42,
                 use_fused: bool = True):
        """
        Args:
            input_size: Size of input vector (from encoder or previous layer)
                       Should match: (batch_size, input_size)
            output_size: Number of LUT nodes (output will be batch_size, output_size * output_dim)
            node_type: LUT node class to use
            node_kwargs: Additional node arguments (should include input_dim and output_dim)
                        Dimension spec: nodes expect (batch_size, output_size, node_input_dim)
            seed: Random seed for reproducible mapping
            use_fused: If True, use fused mapping+forward operation to save memory (default: True)
                      If False, use standard get_mapping() approach (higher memory but simpler)
        """
        self.seed = seed
        self.use_fused = use_fused
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional memory optimization through fused operation.
        
        When use_fused=True (default), uses a custom autograd function that fuses
        the mapping and node forward pass, avoiding storage of intermediate
        (batch_size, output_size, n) activation tensor. This significantly reduces
        memory usage during backpropagation for large batch sizes.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output tensor of shape (batch_size, output_size * output_dim)
        """
        # Validate input dimensions
        self._validate_input_dims(x)
        
        if self.use_fused:
            # Use fused mapping + node forward to avoid storing intermediate activations
            # This is much more memory-efficient for large batches during backprop
            output = RandomMappingFunction.apply(x, self._mapping, self.node, 
                                                  self.output_size, self.n)
        else:
            # Standard approach: call get_mapping then node
            # Higher memory usage but easier to debug
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
        fused_str = "fused" if self.use_fused else "standard"
        return f"input_size={self.input_size}, output_size={self.output_size}, " \
               f"n={self.n}, seed={self.seed}, mapping=random, mode={fused_str}"