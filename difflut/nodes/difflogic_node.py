"""
DiffLogic Node: Differentiable logic gate node.

Implements differentiable 2-input logic gates.
For each output, learns a distribution over all 16 possible 2-input Boolean functions.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..registry import register_node
from .base_node import BaseNode

DEFAULT_DIFFLOGIC_EVAL_MODE: str = "expectation"

BOOLEAN_FUNCTIONS_16 = torch.tensor(
    [
        [0, 0, 0, 0],  # 0:  False
        [1, 0, 0, 0],  # 1:  a AND NOT b
        [0, 1, 0, 0],  # 2:  NOT a AND b
        [1, 1, 0, 0],  # 3:  a NAND b
        [0, 0, 1, 0],  # 4:  a AND NOT b
        [1, 0, 1, 0],  # 5:  b = NOT (a XOR b)
        [0, 1, 1, 0],  # 6:  a XOR b
        [1, 1, 1, 0],  # 7:  a OR b
        [0, 0, 0, 1],  # 8:  a NOR b
        [1, 0, 0, 1],  # 9:  a XNOR b
        [0, 1, 0, 1],  # 10: NOT a
        [1, 1, 0, 1],  # 11: a OR NOT b
        [0, 0, 1, 1],  # 12: NOT b
        [1, 0, 1, 1],  # 13: NOT a OR b
        [0, 1, 1, 1],  # 14: a OR NOT b
        [1, 1, 1, 1],  # 15: True
    ],
    dtype=torch.float32,
)


@register_node("difflogic")
class DiffLogicNode(BaseNode):
    """
    Differentiable Logic Gate Node.

    Takes n input dimensions and for each output, selects 2 inputs.
    Learns a distribution over all 16 possible 2-input Boolean functions.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        init_fn: Optional[Callable[[torch.Tensor], None]] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        regularizers: Optional[Dict[str, Tuple[Callable, float, Dict[str, Any]]]] = None,
        eval_mode: str = DEFAULT_DIFFLOGIC_EVAL_MODE,
    ) -> None:
        """
        Args:
            input_dim: Number of input dimensions (must be >= 2)
            output_dim: Number of output dimensions
            init_fn: Optional initialization function for parameters
            init_kwargs: Keyword arguments for init_fn
            regularizers: Dict of custom regularization functions
            eval_mode: Evaluation mode ('expectation' or 'deterministic')
        """
        if init_kwargs is None:
            init_kwargs = {}
        else:
            init_kwargs = init_kwargs.copy()

        if "node_type" not in init_kwargs:
            init_kwargs["node_type"] = "difflogic"
        if "input_dim" not in init_kwargs:
            if input_dim is not None:
                init_kwargs["input_dim"] = input_dim

        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            regularizers=regularizers,
            init_fn=init_fn,
            init_kwargs=init_kwargs,
        )

        if self.input_dim < 2:
            raise ValueError(
                f"DiffLogicNode requires input_dim >= 2, but got {self.input_dim}. "
                f"The node selects 2 inputs for each output."
            )

        assert eval_mode in {"expectation", "deterministic"}, f"Invalid eval_mode: {eval_mode}"
        self.eval_mode = eval_mode

        # For each output, learn a distribution over 16 Boolean functions
        # Shape: (output_dim, 16)
        self.weights = nn.Parameter(torch.randn(self.output_dim, 16))
        self._apply_init_fn(self.weights, name="weights")

        # For each output, randomly select 2 input indices
        # Shape: (output_dim, 2)
        self.register_buffer("indices", self._get_connections())

        # Register the truth tables as a buffer
        self.register_buffer("truth_tables", BOOLEAN_FUNCTIONS_16)

    def _get_connections(self) -> torch.Tensor:
        """Get random input index pairs for each output."""
        indices = torch.zeros((self.output_dim, 2), dtype=torch.int64)

        for out_idx in range(self.output_dim):
            perm = torch.randperm(self.input_dim)
            indices[out_idx, 0] = perm[0]
            indices[out_idx, 1] = perm[1]

        return indices

    def _compute_boolean_function_output(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        func_idx: int,
    ) -> torch.Tensor:
        """
        Compute the output of a Boolean function for continuous inputs.

        Args:
            a: Tensor of shape (batch_size,) with values in [0, 1]
            b: Tensor of shape (batch_size,) with values in [0, 1]
            func_idx: Integer in [0, 15] specifying which Boolean function

        Returns:
            Tensor of shape (batch_size,) with values in [0, 1]
        """
        truth_table = self.truth_tables[func_idx]

        # Differentiable polynomial interpolation
        # f(a, b) = (1-a)(1-b)*tt[0] + (1-a)*b*tt[1] + a*(1-b)*tt[2] + a*b*tt[3]
        output = (
            (1 - a) * (1 - b) * truth_table[0]
            + (1 - a) * b * truth_table[1]
            + a * (1 - b) * truth_table[2]
            + a * b * truth_table[3]
        )

        return output

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training: probabilistic expectation.

        For each output, computes:
            y_i = Σ_{f=0}^{15} softmax(weights_i)_f * f(a_i, b_i)

        Args:
            x: Input tensor of shape (batch_size, input_dim) with values in [0, 1]

        Returns:
            Output tensor of shape (batch_size, output_dim) with values in [0, 1]
        """
        batch_size = x.shape[0]

        # Compute softmax over Boolean functions for each output
        # Shape: (output_dim, 16)
        probs = torch.softmax(self.weights, dim=-1)

        # Initialize output
        output = torch.zeros(batch_size, self.output_dim, device=x.device, dtype=x.dtype)

        # For each output dimension
        for out_idx in range(self.output_dim):
            # Get the 2 selected input indices
            idx_a, idx_b = self.indices[out_idx]
            a = x[:, idx_a]
            b = x[:, idx_b]

            # Compute weighted sum over all 16 Boolean functions
            for func_idx in range(16):
                func_output = self._compute_boolean_function_output(a, b, func_idx)
                output[:, out_idx] += probs[out_idx, func_idx] * func_output

        return output

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during evaluation.

        Mode 'expectation': Same as training (probabilistic)
        Mode 'deterministic': Select the Boolean function with highest logit

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        if self.eval_mode == "expectation":
            return self.forward_train(x)

        elif self.eval_mode == "deterministic":
            batch_size = x.shape[0]

            # For each output, select the Boolean function with highest weight
            selected_funcs = torch.argmax(self.weights, dim=-1)

            output = torch.zeros(batch_size, self.output_dim, device=x.device, dtype=x.dtype)

            # For each output dimension
            for out_idx in range(self.output_dim):
                # Get the 2 selected input indices
                idx_a, idx_b = self.indices[out_idx]
                a = x[:, idx_a]
                b = x[:, idx_b]

                # Compute output using selected Boolean function
                func_idx = selected_funcs[out_idx].item()
                output[:, out_idx] = self._compute_boolean_function_output(a, b, func_idx)

            return output

        else:
            raise ValueError(f"Unknown eval_mode: {self.eval_mode}")

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization for DiffLogic nodes."""
        return torch.tensor(0.0, device=self.weights.device)

    def extra_repr(self) -> str:
        """Extra information for print/repr."""
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"eval_mode='{self.eval_mode}'"
        )
