import torch
import torch.nn as nn
from typing import Type, Dict, Any, Optional, List
from .base_layer import BaseLUTLayer
from .random_layer import RandomLayer
from ..registry import register_layer

@register_layer("residual")
class ResidualLayer(BaseLUTLayer):
    """
    LUT layer with residual connections using concatenation.
    
    Applies a sequence of random layers with intermediate sizes, then concatenates
    the final output with the original input, and passes through a final random layer.
    
    Example:
        input_size=1000, intermediate_sizes=[200, 100], output_size=800
        1. Apply RandomLayer: 1000 -> 200
        2. Apply RandomLayer: 200 -> 100
        3. Concatenate: [1000, 100] -> 1100
        4. Apply RandomLayer: 1100 -> 800
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int, 
                 node_type: Type[nn.Module],
                 intermediate_sizes: List[int],
                 node_kwargs: Optional[Dict[str, Any]] = None,
                 seed: int = 42):
        """
        Args:
            input_size: Size of input vector
            output_size: Size of output vector  
            node_type: LUT node class to use
            intermediate_sizes: List of intermediate layer sizes (e.g., [200, 100])
            node_kwargs: Additional node arguments (should include input_dim and output_dim)
            seed: Random seed for reproducible mapping
        """
        self.intermediate_sizes = intermediate_sizes
        self.seed = seed
        
        # Initialize parent with final layer dimensions (n will be extracted from created nodes)
        super().__init__(input_size, output_size, node_type, node_kwargs)
        
        # Build the sequence of intermediate layers
        self._build_layers()
    
    def _build_layers(self):
        """
        Build the sequence of random layers:
        1. Intermediate layers (input_size -> intermediate_sizes)
        2. Final layer (input_size + intermediate_sizes[-1] -> output_size)
        """
        self.intermediate_layers = nn.ModuleList()
        
        # Build intermediate layers
        current_size = self.input_size
        for i, intermediate_size in enumerate(self.intermediate_sizes):
            layer = RandomLayer(
                input_size=current_size,
                output_size=intermediate_size,
                node_type=type(self.nodes[0]),  # Use the same node type
                node_kwargs=self._get_node_kwargs(),
                seed=self.seed + i  # Different seed for each layer
            )
            self.intermediate_layers.append(layer)
            current_size = intermediate_size
        
        # Build final layer that takes concatenated input
        # Concatenated size = original input + last intermediate output
        concat_size = self.input_size + self.intermediate_sizes[-1] if self.intermediate_sizes else self.input_size
        self.final_layer = RandomLayer(
            input_size=concat_size,
            output_size=self.output_size,
            node_type=type(self.nodes[0]),
            node_kwargs=self._get_node_kwargs(),
            seed=self.seed + len(self.intermediate_sizes)
        )
    
    def _get_node_kwargs(self) -> Dict[str, Any]:
        """Extract node kwargs from the first node."""
        # Get the constructor arguments from the first node
        # This is a simplified approach - you might need to adjust based on your node types
        node_kwargs = {}
        first_node = self.nodes[0]
        
        # Include input_dim and output_dim (these are required)
        node_kwargs['input_dim'] = first_node.input_dim
        node_kwargs['output_dim'] = first_node.output_dim
        
        # Common attributes that might be passed as kwargs
        for attr in ['temperature', 'init_scale', 'use_bias', 'activation', 'use_surrogate', 'regularizers', 'init_fn']:
            if hasattr(first_node, attr):
                node_kwargs[attr] = getattr(first_node, attr)
        
        return node_kwargs
    
    def get_mapping(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method is required by BaseLUTLayer but not used directly.
        The actual forward pass is overridden.
        """
        # Not used in this implementation
        raise NotImplementedError("ResidualLayer uses custom forward pass")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual layer.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Store original input for residual connection
        residual = x
        
        # Apply intermediate layers sequentially
        current = x
        for layer in self.intermediate_layers:
            current = layer(current)
        
        # Concatenate with original input
        concatenated = torch.cat([residual, current], dim=1)
        
        # Apply final layer
        output = self.final_layer(concatenated)
        
        return output
    
    def regularization(self) -> torch.Tensor:
        """Compute regularization for all layers."""
        reg = torch.tensor(0.0, device=next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
        # Add regularization from intermediate layers
        for layer in self.intermediate_layers:
            if hasattr(layer, 'regularization'):
                reg = reg + layer.regularization()
        
        # Add regularization from final layer
        if hasattr(self.final_layer, 'regularization'):
            reg = reg + self.final_layer.regularization()
        
        # Average over all layers
        total_layers = len(self.intermediate_layers) + 1
        return reg / total_layers if total_layers > 0 else reg
    
    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return f"input_size={self.input_size}, output_size={self.output_size}, " \
               f"intermediate_sizes={self.intermediate_sizes}, n={self.n}, " \
               f"seed={self.seed}, mapping=residual"
