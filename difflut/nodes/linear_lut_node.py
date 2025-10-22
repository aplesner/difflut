import torch
import torch.nn as nn
import itertools
from typing import Optional, Callable
from .base_node import BaseNode
from ..registry import register_node

@register_node("linear_lut")
class LinearLUTNode(BaseNode):
    """
    Linear LUT Node with differentiable training and hard decisions for eval.
    Uses autograd for gradient computation.
    """
    
    def __init__(self, 
                 input_dim: int = None,
                 output_dim: int = None,
                 init_fn: Optional[Callable] = None,
                 init_kwargs: dict = None,
                 regularizers: dict = None):
        """
        Args:
            input_dim: Number of inputs (e.g., 6)
            output_dim: Number of outputs (e.g., 1)
            init_fn: Optional initialization function. Should take (param: torch.Tensor, **kwargs)
            init_kwargs: Keyword arguments for init_fn
            regularizers: Dict of custom regularization functions
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, regularizers=regularizers, init_fn=init_fn, init_kwargs=init_kwargs)
        
        # Initialize weights with default values, then apply init_fn if provided
        self.weights = nn.Parameter(torch.randn(self.num_inputs, self.num_outputs) * 0.1)
        self._apply_init_fn(self.weights, name="weights")

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training: Use sigmoid for differentiability.
        Args:
            x: Input tensor (batch_size, num_inputs)
        """
        # Linear transformation + sigmoid
        z = torch.matmul(x, self.weights)  # (batch_size, num_outputs)
        output = torch.sigmoid(z)
        
        return output

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Discretize by applying Heaviside at 0.5 to forward_train output.
        This makes it behave like a real LUT with binary outputs.
        """
        # Compute same as forward_train (linear + sigmoid)
        z = torch.matmul(x, self.weights)
        
        # Discretize: Heaviside at 0.0 since forward_train output is in R
        output = (z >= 0.0).float()
        
        return output

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization by default."""
        return torch.tensor(0.0, device=self.weights.device)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"