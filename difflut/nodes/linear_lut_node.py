import itertools
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..registry import register_node
from .base_node import BaseNode


@register_node("linear_lut")
class LinearLUTNode(BaseNode):
    """
    Linear LUT Node with differentiable training and hard decisions for eval.
    Uses autograd for gradient computation.

    Processes 2D tensors: (batch_size, input_dim) → (batch_size, output_dim)
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        init_fn: Optional[Callable[[torch.Tensor], None]] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        regularizers: Optional[Dict[str, Tuple[Callable, float, Dict[str, Any]]]] = None,
    ) -> None:
        """
        Args:
            input_dim: Number of inputs (e.g., 6)
            output_dim: Number of outputs (e.g., 1)
            init_fn: Optional initialization function. Should take (param: torch.Tensor, **kwargs)
            init_kwargs: Keyword arguments for init_fn
            regularizers: Dict of custom regularization functions
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            regularizers=regularizers,
            init_fn=init_fn,
            init_kwargs=init_kwargs,
        )

        # Initialize weights
        # Shape: (input_dim, output_dim)
        self.weights = nn.Parameter(torch.randn(self.num_inputs, self.num_outputs) * 0.1)
        self._apply_init_fn(self.weights, name="weights")

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training: Use sigmoid for differentiability.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size, input_dim = x.shape

        # Matrix multiplication
        # x: (batch_size, input_dim)
        # weights: (input_dim, output_dim)
        # output: (batch_size, output_dim)
        z = torch.matmul(x, self.weights)
        output = torch.sigmoid(z)

        return output

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Binarize output via Heaviside.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size, input_dim = x.shape

        # Matrix multiplication
        z = torch.matmul(x, self.weights)
        output = (torch.sigmoid(z) >= 0.5).float()

        return output

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization by default."""
        return torch.tensor(0.0, device=self.weights.device)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"
