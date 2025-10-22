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
                 input_dim: list = None,
                 output_dim: list = None,
                 init_fn=None,
                 init_kwargs: dict = None,
                 regularizers: dict = None):
        """
        Args:
            input_dim: Input dimensions as list (e.g., [6])
            output_dim: Output dimensions as list (e.g., [1])
            init_fn: Optional initialization function
            init_kwargs: Optional dict of kwargs for initializer
            regularizers: Dict of regularization (e.g., {"l1": [0.01], "spectral": [0.001, {"num": 200}]})
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, 
                         init_fn=init_fn, init_kwargs=init_kwargs, regularizers=regularizers)
        
        # Initialize weights (init will be applied in parent)
        self.weights = nn.Parameter(torch.randn(self.num_inputs, self.num_outputs) * 0.1)

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training: Use sigmoid for differentiability.
        Args:
            x: Input tensor (batch_size, num_inputs) or (batch_size, 1, num_inputs)
        """
        x = self._prepare_input(x)
        
        # Linear transformation + sigmoid
        z = torch.matmul(x, self.weights)  # (batch_size, num_outputs)
        output = torch.sigmoid(z)
        
        return self._prepare_output(output)

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Discretize by applying Heaviside at 0.5 to forward_train output.
        This makes it behave like a real LUT with binary outputs.
        """
        x = self._prepare_input(x)
        
        # Compute same as forward_train (linear + sigmoid)
        z = torch.matmul(x, self.weights)
        
        # Discretize: Heaviside at 0.0 since forward_train output is in R
        output = (z >= 0.0).float()
        
        return self._prepare_output(output)

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization by default."""
        return torch.tensor(0.0, device=self.weights.device)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"