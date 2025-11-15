import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..registry import register_node
from ..utils.warnings import warn_default_value
from .base_node import BaseNode

# Default maximum polynomial degree for PolyLUT nodes
DEFAULT_POLYLUT_DEGREE: int = 3


@register_node("polylut")
class PolyLUTNode(BaseNode):
    """
    Polynomial LUT Node that computes multivariate polynomials up to degree D.
    Uses autograd for gradient computation.

    Processes 2D tensors: (batch_size, input_dim) → (batch_size, output_dim)
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        degree: int = DEFAULT_POLYLUT_DEGREE,
        init_fn: Optional[Callable[[torch.Tensor], None]] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        regularizers: Optional[Dict[str, Tuple[Callable, float, Dict[str, Any]]]] = None,
    ) -> None:
        """
        Polynomial LUT Node that computes multivariate polynomials up to degree D.

        Parameters:
        - input_dim: Optional[int], Number of inputs (e.g., 6), (default: None)
        - output_dim: Optional[int], Number of outputs (e.g., 1), (default: None)
        - degree: int, Maximum degree of polynomial terms, (default: 3)
        - init_fn: Optional[Callable], Initialization function for parameters, (default: None)
        - init_kwargs: Optional[Dict[str, Any]], Keyword arguments for init_fn, (default: None)
        - regularizers: Optional[Dict], Custom regularization functions, (default: None)
        """
        # Generate monomial combinations BEFORE calling super().__init__
        # This is needed because residual initialization requires monomial_combinations
        # We need to determine input_dim first (use default if not provided)
        from .base_node import DEFAULT_NODE_INPUT_DIM

        temp_input_dim = input_dim if input_dim is not None else DEFAULT_NODE_INPUT_DIM
        monomial_combinations = PolyLUTNode._generate_monomial_combinations(temp_input_dim, degree)

        # Prepare init_kwargs with required parameters for residual initialization
        if init_kwargs is None:
            init_kwargs = {}
        else:
            init_kwargs = init_kwargs.copy()  # Make a copy to avoid modifying the original

        # Always set monomial_combinations, even if it exists but is None
        if (
            "monomial_combinations" not in init_kwargs
            or init_kwargs["monomial_combinations"] is None
        ):
            init_kwargs["monomial_combinations"] = monomial_combinations

        # Add input_dim to init_kwargs if not already present (needed for residual_init)
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
        self.degree = degree

        # Store monomial combinations generated earlier
        self.monomial_combinations = monomial_combinations
        self.num_monomials = len(self.monomial_combinations)

        # Store as buffer for efficient computation
        self.register_buffer(
            "exponent_matrix", torch.tensor(self.monomial_combinations, dtype=torch.float32)
        )

        # Initialize weights for polynomial coefficients
        # Shape: (num_monomials, num_outputs)
        self.weights = nn.Parameter(torch.randn(self.num_monomials, self.num_outputs) * 0.1)

        self._apply_init_fn(self.weights, name="weights")

    @staticmethod
    def _generate_monomial_combinations(num_inputs: int, degree: int) -> list:
        """Generate all monomial combinations up to given degree."""
        combinations = []

        # Include constant term
        combinations.append(tuple([0] * num_inputs))

        # Generate monomials for degrees 1 to D
        for d in range(1, degree + 1):
            for exponents in PolyLUTNode._integer_compositions(num_inputs, d):
                combinations.append(exponents)

        return combinations

    @staticmethod
    def _integer_compositions(n: int, d: int):
        """Generate all n-tuples of non-negative integers that sum to d."""
        if n == 1:
            yield (d,)
        else:
            for i in range(d + 1):
                for comp in PolyLUTNode._integer_compositions(n - 1, d - i):
                    yield (i,) + comp

    def _compute_monomials(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute all monomial values for input x.
        Args:
            x: shape [batch_size, num_inputs] with values in [0,1]
        Returns:
            monomial matrix of shape [batch_size, num_monomials]
        """
        # Vectorized monomial computation
        x_expanded = x.unsqueeze(1)  # [batch, 1, num_inputs]
        exponents = self.exponent_matrix.unsqueeze(0)  # [1, num_monomials, num_inputs]

        # Handle zero exponents (avoid 0^0)
        mask = exponents > 0
        safe_x = torch.where(mask, x_expanded, torch.ones_like(x_expanded))

        # Compute x^exponents
        monomials = (safe_x**exponents).prod(dim=-1)  # [batch, num_monomials]

        return monomials

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training: polynomial transform + sigmoid activation.

        Args:
            x: Input tensor (batch_size, num_inputs)
        Returns:
            Output tensor (batch_size, num_outputs)
        """
        # Ensure input is on the same device as parameters
        x = x.to(self.weights.device)
        
        batch_size, input_dim = x.shape

        # Compute all monomials: (batch_size, num_monomials)
        monomials = self._compute_monomials(x)

        # Linear combination of monomials
        # monomials: (batch_size, num_monomials)
        # weights: (num_monomials, num_outputs)
        # output: (batch_size, num_outputs)
        z = torch.matmul(monomials, self.weights)
        output = torch.sigmoid(z)

        return output

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Discretized output via Heaviside at 0.5.

        Args:
            x: Input tensor (batch_size, num_inputs)
        Returns:
            Output tensor (batch_size, num_outputs)
        """
        # Ensure input is on the same device as parameters
        x = x.to(self.weights.device)
        
        batch_size, input_dim = x.shape

        # Compute all monomials: (batch_size, num_monomials)
        monomials = self._compute_monomials(x)

        # Linear combination
        z = torch.matmul(monomials, self.weights)
        output = (torch.sigmoid(z) >= 0.5).float()

        return output

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization by default."""
        return torch.tensor(0.0, device=self.weights.device)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"degree={self.degree}, num_monomials={self.num_monomials}"
        )
