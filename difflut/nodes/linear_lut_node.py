import itertools
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..registry import register_node
from ..utils.warnings import warn_default_value
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
        Linear LUT Node with differentiable training and hard decisions for eval.

        Parameters:
        - input_dim: Optional[int], Number of inputs (e.g., 6), (default: None)
        - output_dim: Optional[int], Number of outputs (e.g., 1), (default: None)
        - init_fn: Optional[Callable], Initialization function for parameters, (default: None)
        - init_kwargs: Optional[Dict[str, Any]], Keyword arguments for init_fn, (default: None)
        - regularizers: Optional[Dict], Custom regularization functions, (default: None)
        """
        # Prepare init_kwargs with required parameters for residual initialization
        if init_kwargs is None:
            init_kwargs = {}
        else:
            init_kwargs = init_kwargs.copy()  # Make a copy to avoid modifying the original

        # Add input_dim to init_kwargs if not already present (needed for residual_init)
        if "input_dim" not in init_kwargs:
            # Use the input_dim parameter, or default if not provided
            if input_dim is not None:
                init_kwargs["input_dim"] = input_dim

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

        # Initialize bias (needed for residual initialization to work properly)
        # Shape: (output_dim,)
        self.bias = nn.Parameter(torch.zeros(self.num_outputs))

        # Apply initialization with param_name for weights
        if self.init_fn is not None:
            weights_init_kwargs = self.init_kwargs.copy()
            weights_init_kwargs["param_name"] = "weights"
            try:
                self.init_fn(self.weights, **weights_init_kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"Initialization of 'weights' failed with error: {e}. "
                    f"Check that init_fn is compatible with the parameter and init_kwargs are correct."
                )

        # Apply initialization with param_name for bias
        if self.init_fn is not None:
            bias_init_kwargs = self.init_kwargs.copy()
            bias_init_kwargs["param_name"] = "bias"
            try:
                self.init_fn(self.bias, **bias_init_kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"Initialization of 'bias' failed with error: {e}. "
                    f"Check that init_fn is compatible with the parameter and init_kwargs are correct."
                )

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training: Use sigmoid for differentiability.

        Parameters:
        - x: torch.Tensor, Input tensor of shape (batch_size, input_dim)
        """
        batch_size, input_dim = x.shape

        # Matrix multiplication
        # x: (batch_size, input_dim)
        # weights: (input_dim, output_dim)
        # output: (batch_size, output_dim)
        z = torch.matmul(x, self.weights) + self.bias
        output = torch.sigmoid(z)

        return output

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Binarize output via Heaviside.

        Parameters:
        - x: torch.Tensor, Input tensor of shape (batch_size, input_dim)
        """
        batch_size, input_dim = x.shape

        # Matrix multiplication
        z = torch.matmul(x, self.weights) + self.bias
        output = (torch.sigmoid(z) >= 0.5).float()

        return output

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization by default."""
        return torch.tensor(0.0, device=self.weights.device)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"
