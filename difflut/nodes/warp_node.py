"""
WARP-LUTs Node: Walsh-Assisted Relaxation for Probabilistic Look Up Tables.

Implements Walsh-Hadamard decomposition for Boolean functions with differentiable
relaxation for training. Uses signed basis {-1,+1} for inputs and Walsh coefficients
to represent arbitrary Boolean functions.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..registry import register_node
from ..utils.warnings import warn_default_value
from .base_node import BaseNode

# Default temperature for sigmoid scaling in WARP nodes
DEFAULT_WARP_TEMPERATURE: float = 1.0
# Default evaluation mode for WARP nodes
# Options: 'sign', 'threshold'
DEFAULT_WARP_EVAL_MODE: str = "sign"


@register_node("warp")
class WARPNode(BaseNode):
    """
    WARP-LUTs: Walsh-Assisted Relaxation for Probabilistic Look Up Tables.

    Uses Walsh-Hadamard decomposition to represent Boolean functions:
        f(x) = sign( Σ_{S ⊆ {1,...,n}} c_S · Π_{i ∈ S} B(x_i) )

    Key features:
    - Input transformation: B̃(x) = 2x - 1  (maps [0,1] → [-1,1])
    - Walsh coefficients c_S for all 2^n subsets S
    - Forward (training): σ( [Σ c_S · Π B̃(x_i)] / τ )
    - Forward (eval): sign( Σ c_S · Π B̃(x_i) )

    For 2 inputs: l(a,b) = c₀ + c₁·B̃(a) + c₂·B̃(b) + c₃·B̃(a)·B̃(b)

    Weights: Walsh coefficients c ∈ ℝ^(2^n) for each output
    Processes 2D tensors: (batch_size, input_dim) -> (batch_size, output_dim)
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        init_fn: Optional[Callable[[torch.Tensor], None]] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        regularizers: Optional[Dict[str, Tuple[Callable, float, Dict[str, Any]]]] = None,
        temperature: float = DEFAULT_WARP_TEMPERATURE,
        eval_mode: str = DEFAULT_WARP_EVAL_MODE,
    ) -> None:
        """
        Args:
            input_dim: Number of inputs (e.g., 2 for 2-input gates)
            output_dim: Number of outputs (e.g., 1)
            init_fn: Optional initialization function. Should take (param: torch.Tensor, **kwargs)
            init_kwargs: Keyword arguments for init_fn
            regularizers: Dict of custom regularization functions
            temperature: Temperature τ for sigmoid scaling during training
            eval_mode: Evaluation mode ('sign' or 'threshold')
        """
        # Prepare init_kwargs with required parameters for residual initialization
        if init_kwargs is None:
            init_kwargs = {}
        else:
            init_kwargs = init_kwargs.copy()  # Make a copy to avoid modifying the original

        # Add node_type and input_dim to init_kwargs if not already present (needed for residual_init)
        if "node_type" not in init_kwargs:
            init_kwargs["node_type"] = "warp"
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

        self.register_buffer("temperature", torch.tensor(float(temperature)))
        assert eval_mode in {"sign", "threshold"}, f"Invalid eval_mode: {eval_mode}"
        self.eval_mode = eval_mode

        # Number of Walsh coefficients: 2^input_dim
        self.num_coefficients = 2**self.input_dim

        # Store Walsh coefficients
        # Shape: (num_coefficients, output_dim)
        self.coefficients = nn.Parameter(torch.randn(self.num_coefficients, self.output_dim) * 0.1)
        self._apply_init_fn(self.coefficients, name="coefficients")

        # Precompute subset masks for Walsh basis functions
        # subset_masks[i] is a list of input indices that form subset i
        self.subset_masks = self._compute_subset_masks()

    def _compute_subset_masks(self) -> list:
        """
        Precompute which inputs participate in each Walsh basis function.

        For each subset index i ∈ {0, ..., 2^n - 1}, determine which input
        indices are in the subset (based on binary representation of i).

        Returns:
            List of lists: subset_masks[i] = [j1, j2, ...] for subset i
        """
        masks = []
        for subset_idx in range(self.num_coefficients):
            # Find which bits are set in subset_idx
            indices = [j for j in range(self.input_dim) if (subset_idx >> j) & 1]
            masks.append(indices)
        return masks

    def _signed_basis_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform inputs from [0,1] to signed basis [-1,+1].

        B̃(x) = 2x - 1

        Args:
            x: Input tensor of shape (batch_size, input_dim) in [0, 1]

        Returns:
            Transformed tensor of shape (batch_size, input_dim) in [-1, 1]
        """
        return 2.0 * x - 1.0

    def _compute_walsh_basis_functions(self, x_signed: torch.Tensor) -> torch.Tensor:
        """
        Compute all Walsh basis functions for signed inputs.

        For each subset S, compute: Π_{i ∈ S} B̃(x_i)

        Args:
            x_signed: Signed input tensor of shape (batch_size, input_dim) in [-1, 1]

        Returns:
            Walsh basis tensor of shape (batch_size, num_coefficients)
        """
        batch_size = x_signed.shape[0]
        walsh_basis = torch.ones(
            batch_size, self.num_coefficients, device=x_signed.device, dtype=x_signed.dtype
        )

        # Compute each Walsh basis function
        for subset_idx in range(self.num_coefficients):
            indices = self.subset_masks[subset_idx]
            if len(indices) > 0:
                # Product over selected inputs
                for j in indices:
                    walsh_basis[:, subset_idx] *= x_signed[:, j]

        return walsh_basis

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training: differentiable relaxation with sigmoid.

        Computes: f̃(x) = σ( [Σ_{S} c_S · Π_{i∈S} B̃(x_i)] / τ )

        Args:
            x: Input tensor of shape (batch_size, input_dim) in [0, 1]

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Transform to signed basis: (batch_size, input_dim)
        x_signed = self._signed_basis_transform(x)

        # Compute Walsh basis functions: (batch_size, num_coefficients)
        walsh_basis = self._compute_walsh_basis_functions(x_signed)

        # Weighted sum with Walsh coefficients
        # walsh_basis: (batch_size, num_coefficients)
        # coefficients: (num_coefficients, output_dim)
        # logits: (batch_size, output_dim)
        logits = torch.matmul(walsh_basis, self.coefficients)

        # Apply temperature-scaled sigmoid
        output = torch.sigmoid(logits / self.temperature)

        return output

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during evaluation: deterministic output.

        Mode 'sign': f(x) = sign( Σ_{S} c_S · Π_{i∈S} B̃(x_i) )
                     Returns {0, 1} using (sign(·) + 1) / 2

        Mode 'threshold': Same as sign but with Heaviside threshold

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim) in {0, 1}
        """
        # Transform to signed basis: (batch_size, input_dim)
        x_signed = self._signed_basis_transform(x)

        # Compute Walsh basis functions: (batch_size, num_coefficients)
        walsh_basis = self._compute_walsh_basis_functions(x_signed)

        # Weighted sum with Walsh coefficients
        # logits: (batch_size, output_dim)
        logits = torch.matmul(walsh_basis, self.coefficients)

        if self.eval_mode == "sign":
            # Apply sign function: -1 for negative, +1 for positive
            # Convert to {0, 1}: (sign(x) + 1) / 2
            output = (torch.sign(logits) + 1.0) / 2.0
        elif self.eval_mode == "threshold":
            # Threshold at 0: output 1 if logit > 0, else 0
            output = (logits > 0.0).float()
        else:
            raise ValueError(f"Unknown eval_mode: {self.eval_mode}")

        return output

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization for WARP nodes."""
        return torch.tensor(0.0, device=self.coefficients.device)

    def extra_repr(self) -> str:
        """Extra information for print/repr."""
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"num_coefficients={self.num_coefficients}, "
            f"temperature={self.temperature.item():.3f}, eval_mode='{self.eval_mode}'"
        )
