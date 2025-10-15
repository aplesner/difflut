import torch
import torch.nn as nn
import itertools
from .base_node import BaseNode
from ..registry import register_node

@register_node("linear_lut")
class LinearLUTNode(BaseNode):
    """
    Linear LUT Node with differentiable training and hard decisions for eval.
    Uses autograd for gradient computation.
    """
    
    def __init__(self, 
                 num_inputs: int, 
                 output_dim: int = 1,
                 init_fn=None,
                 regularizers: dict = None):
        """
        Args:
            num_inputs: Number of inputs to the LUT
            output_dim: Number of output values
            init_fn: Optional initialization function
            regularizers: Dict of custom regularization functions
        """
        super().__init__(num_inputs=num_inputs, regularizers=regularizers)
        self.output_dim = output_dim
        
        # Initialize weights
        if init_fn:
            self.weights = nn.Parameter(init_fn((num_inputs, output_dim)))
        else:
            self.weights = nn.Parameter(torch.randn(num_inputs, output_dim) * 0.1)

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training: Use sigmoid for differentiability.
        Args:
            x: Input tensor (batch_size, num_inputs) or (batch_size, 1, num_inputs)
        """
        # Handle different input dimensions
        if x.dim() == 3:
            x = x.squeeze(1)  # Remove middle dimension
        
        # Linear transformation + sigmoid
        z = torch.matmul(x, self.weights)  # (batch_size, output_dim)
        output = torch.sigmoid(z)
        
        # Squeeze if output_dim is 1
        if self.output_dim == 1:
            output = output.squeeze(-1)
        
        return output

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Use step function for binary output.
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        
        z = torch.matmul(x, self.weights)
        output = (z >= 0).float()
        
        if self.output_dim == 1:
            output = output.squeeze(-1)
        
        return output

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization by default."""
        return torch.tensor(0.0, device=self.weights.device)

    def extra_repr(self) -> str:
        return f"num_inputs={self.num_inputs}, output_dim={self.output_dim}"