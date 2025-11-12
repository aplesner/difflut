"""
DiffLogic Node: Monomial-based differentiable logic gate node.

Implements a monomial-based formulation for n-input differentiable logic gates.
Each Boolean function is represented using Reed-Muller expansion with monomials.
"""

import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..registry import register_node
from ..utils.warnings import warn_default_value
from .base_node import BaseNode

# Default temperature for softmax over Boolean functions
DEFAULT_DIFFLOGIC_TEMPERATURE: float = 1.0
# Default evaluation mode for DiffLogic nodes
# Options: 'expectation', 'deterministic'
DEFAULT_DIFFLOGIC_EVAL_MODE: str = "expectation"


@register_node("difflogic")
class DiffLogicNode(BaseNode):
    """
    Differentiable Logic Gate Node with monomial-based formulation.

    For n inputs, learns a distribution over all 2^(2^n) Boolean functions
    using softmax parameterization. Each Boolean function is computed via
    its Reed-Muller monomial expansion.

    Forward (training): Weighted sum of all Boolean function expectations
    Forward (eval): Can use 'expectation' or 'deterministic' mode

    Weights: Logits z ∈ ℝ^(2^(2^n)) over Boolean functions
    Processes 2D tensors: (batch_size, input_dim) -> (batch_size, output_dim)

    Warning: Complexity grows as 2^(2^n):
        n=1: 4 functions
        n=2: 16 functions
        n=3: 256 functions (⚠️ warning issued)
        n=4: 65,536 functions (❌ error - too large)
        n=5+: infeasible
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        init_fn: Optional[Callable[[torch.Tensor], None]] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        regularizers: Optional[Dict[str, Tuple[Callable, float, Dict[str, Any]]]] = None,
        temperature: float = DEFAULT_DIFFLOGIC_TEMPERATURE,
        eval_mode: str = DEFAULT_DIFFLOGIC_EVAL_MODE,
    ) -> None:
        """
        Args:
            input_dim: Number of inputs (e.g., 2 for 2-input logic gates)
            output_dim: Number of outputs (e.g., 1)
            init_fn: Optional initialization function. Should take (param: torch.Tensor, **kwargs)
            init_kwargs: Keyword arguments for init_fn
            regularizers: Dict of custom regularization functions
            temperature: Temperature for softmax over Boolean functions
            eval_mode: Evaluation mode ('expectation' or 'deterministic')
        """
        # Prepare init_kwargs with required parameters for residual initialization
        if init_kwargs is None:
            init_kwargs = {}
        else:
            init_kwargs = init_kwargs.copy()  # Make a copy to avoid modifying the original

        # Add node_type and input_dim to init_kwargs if not already present (needed for residual_init)
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

        # Validate input dimension
        if self.input_dim is not None:
            if self.input_dim >= 3:
                warnings.warn(
                    f"DiffLogicNode with input_dim={self.input_dim} requires "
                    f"{2 ** (2 ** self.input_dim)} Boolean functions. "
                    f"This is computationally expensive and may cause memory issues. "
                    f"(n=3 requires 256 functions, n=4 requires 65,536 functions, "
                    f"n=5 requires ~4.3 billion functions). "
                    f"Consider using input_dim=2 (16 functions) for better efficiency, "
                    f"or tree-based compositions of 2-input nodes.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self.register_buffer("temperature", torch.tensor(float(temperature)))
        assert eval_mode in {"expectation", "deterministic"}, f"Invalid eval_mode: {eval_mode}"
        self.eval_mode = eval_mode

        # Number of Boolean functions: 2^(2^input_dim)
        self.num_functions = 2 ** (2**self.input_dim)

        # Number of monomials: 2^input_dim
        self.num_monomials = 2**self.input_dim

        # Store logits over Boolean functions
        # Shape: (num_functions, output_dim)
        self.logits = nn.Parameter(torch.zeros(self.num_functions, self.output_dim))
        self._apply_init_fn(self.logits, name="logits")

        # Precompute monomial coefficient matrix
        # Shape: (num_functions, num_monomials)
        # coefficients[i, k] = c_ik where g_i(a) = Σ_k c_ik * monomial_k(a)
        self.register_buffer("coefficients", self._compute_monomial_coefficients())

        # Precompute which input indices are used in each monomial
        # monomial_masks[k] = list of input indices j where bit_j(k) = 1
        self.monomial_masks = self._compute_monomial_masks()

    def _compute_monomial_coefficients(self) -> torch.Tensor:
        """
        Compute the monomial coefficient matrix for all Boolean functions.

        For each Boolean function i ∈ {0, ..., 2^(2^n)-1}:
            g_i(x) = Σ_{k=0}^{2^n-1} c_ik * monomial_k(x)

        The coefficients c_ik are determined by the Reed-Muller expansion
        of the Boolean function i.

        Returns:
            Tensor of shape (num_functions, num_monomials) with values in {0, 1}
        """
        n = self.input_dim
        num_functions = 2 ** (2**n)
        num_monomials = 2**n

        coefficients = torch.zeros(num_functions, num_monomials, dtype=torch.float32)

        # For each Boolean function i
        for func_idx in range(num_functions):
            # The function is defined by its truth table (func_idx in binary)
            # truth_table[j] = output of function for input pattern j
            truth_table = torch.tensor(
                [(func_idx >> j) & 1 for j in range(num_monomials)], dtype=torch.float32
            )

            # Compute Reed-Muller coefficients via Möbius transform
            # This is the ANF (Algebraic Normal Form) representation
            coeffs = truth_table.clone()

            # Apply Möbius transform: coefficient computation
            # For each subset S (represented by k), compute XOR over supersets
            for i in range(n):
                bit = 1 << i
                for k in range(num_monomials):
                    if k & bit == 0:
                        # XOR operation (mod 2): we use sum and then mod 2
                        coeffs[k | bit] = (coeffs[k | bit] + coeffs[k]) % 2

            coefficients[func_idx] = coeffs

        return coefficients

    def _compute_monomial_masks(self) -> list:
        """
        Precompute which input indices are involved in each monomial.

        For monomial k, return the list of input indices j where bit_j(k) = 1.

        Returns:
            List of lists: monomial_masks[k] = [j1, j2, ...] for monomial k
        """
        masks = []
        for k in range(self.num_monomials):
            # Find which bits are set in k
            indices = [j for j in range(self.input_dim) if (k >> j) & 1]
            masks.append(indices)
        return masks

    def _compute_monomials(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute all monomial values for inputs x.

        For each monomial k, compute:
            monomial_k(x) = Π_{j: bit_j(k)=1} x_j

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tensor of shape (batch_size, num_monomials)
        """
        batch_size = x.shape[0]
        monomials = torch.ones(batch_size, self.num_monomials, device=x.device, dtype=x.dtype)

        # Compute each monomial
        for k in range(self.num_monomials):
            indices = self.monomial_masks[k]
            if len(indices) > 0:
                # Product over selected inputs
                for j in indices:
                    monomials[:, k] *= x[:, j]

        return monomials

    def _compute_boolean_functions(self, monomials: torch.Tensor) -> torch.Tensor:
        """
        Compute all Boolean function expectations from monomials.

        For each Boolean function i:
            g_i(x) = Σ_{k=0}^{2^n-1} c_ik * monomial_k(x)

        Args:
            monomials: Tensor of shape (batch_size, num_monomials)

        Returns:
            Tensor of shape (batch_size, num_functions)
        """
        # Matrix multiply: (batch_size, num_monomials) @ (num_monomials, num_functions)
        # Result: (batch_size, num_functions)
        return torch.matmul(monomials, self.coefficients.T)

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training: probabilistic expectation.

        Computes weighted sum over all Boolean functions:
            a' = Σ_i [softmax(z/T)_i · g_i(a)]

        Where g_i is computed via monomial expansion.

        Args:
            x: Input tensor of shape (batch_size, input_dim) in [0, 1]

        Returns:
            Output tensor of shape (batch_size, output_dim) in [0, 1]
        """
        batch_size = x.shape[0]

        # Compute all monomials: (batch_size, num_monomials)
        monomials = self._compute_monomials(x)

        # Compute all Boolean function expectations: (batch_size, num_functions)
        bool_funcs = self._compute_boolean_functions(monomials)

        # Apply temperature-scaled softmax to logits: (num_functions, output_dim)
        probs = torch.softmax(self.logits / self.temperature, dim=0)

        # Weighted sum over Boolean functions
        # bool_funcs: (batch_size, num_functions, 1)
        # probs: (1, num_functions, output_dim)
        # Result: (batch_size, output_dim)
        bool_funcs_expanded = bool_funcs.unsqueeze(-1)  # (batch_size, num_functions, 1)
        probs_expanded = probs.unsqueeze(0)  # (1, num_functions, output_dim)

        output = (bool_funcs_expanded * probs_expanded).sum(dim=1)  # (batch_size, output_dim)

        # Clamp to [0, 1] to handle numerical precision issues
        output = torch.clamp(output, 0.0, 1.0)

        return output

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during evaluation.

        Mode 'expectation': Same as training (probabilistic)
        Mode 'deterministic': Use argmax Boolean function

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim) in [0, 1]
        """
        if self.eval_mode == "expectation":
            # Same as training
            return self.forward_train(x)

        elif self.eval_mode == "deterministic":
            batch_size = x.shape[0]

            # Compute all monomials: (batch_size, num_monomials)
            monomials = self._compute_monomials(x)

            # Compute all Boolean function expectations: (batch_size, num_functions)
            bool_funcs = self._compute_boolean_functions(monomials)

            # For each output, select the Boolean function with highest logit
            # argmax over functions: (output_dim,)
            selected_funcs = torch.argmax(self.logits, dim=0)  # (output_dim,)

            # Gather selected Boolean function outputs
            # bool_funcs: (batch_size, num_functions)
            # selected_funcs: (output_dim,)
            output = torch.zeros(batch_size, self.output_dim, device=x.device, dtype=x.dtype)
            for out_idx in range(self.output_dim):
                func_idx = selected_funcs[out_idx]
                output[:, out_idx] = bool_funcs[:, func_idx]

            # Clamp to [0, 1] to handle numerical precision issues
            output = torch.clamp(output, 0.0, 1.0)

            return output

        else:
            raise ValueError(f"Unknown eval_mode: {self.eval_mode}")

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization for DiffLogic nodes."""
        return torch.tensor(0.0, device=self.logits.device)

    def extra_repr(self) -> str:
        """Extra information for print/repr."""
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"num_functions={self.num_functions}, num_monomials={self.num_monomials}, "
            f"temperature={self.temperature.item():.3f}, eval_mode='{self.eval_mode}'"
        )
