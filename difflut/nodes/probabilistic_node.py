import itertools
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..registry import register_node
from ..utils.cuda_utils import should_use_cuda_from_tensor
from ..utils.warnings import CUDAWarning, DefaultValueWarning, warn_default_value
from .base_node import BaseNode
from .cuda import is_cuda_available

# ==================== Default Values ====================
# Module-level constants for all default parameters
# Used for initialization and warning messages

DEFAULT_PROBABILISTIC_TEMPERATURE: float = 1.0
"""Default temperature for probabilistic sigmoid scaling."""

DEFAULT_PROBABILISTIC_EVAL_MODE: str = "expectation"
"""Default evaluation mode for probabilistic nodes. Options: 'expectation', 'deterministic', 'threshold'."""

DEFAULT_PROBABILISTIC_USE_CUDA: bool = True
"""Default flag for using CUDA kernels in probabilistic nodes."""

# Try to import the compiled CUDA extension
try:
    import probabilistic_cuda as _probabilistic_cuda_module

    _CUDA_EXT_AVAILABLE = True
except ImportError:
    _CUDA_EXT_AVAILABLE = False
    _probabilistic_cuda_module = None
    warnings.warn(
        "CUDA extension 'probabilistic_cuda' not available. ProbabilisticNode will use slower CPU fallback. "
        "For better performance, compile the CUDA extension using: "
        "'cd difflut && python setup.py install'. "
        "To suppress this warning: warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut.nodes.probabilistic_node')",
        RuntimeWarning,
        stacklevel=2,
    )


class ProbabilisticFunction(torch.autograd.Function):
    """
    PyTorch autograd function wrapper for Probabilistic CUDA kernels.
    Processes 2D tensors.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        raw_weights: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """
        Forward pass using CUDA kernel.

        Args:
            input: (batch_size, input_dim) float tensor in [0, 1]
            raw_weights: (output_dim, 2^input_dim) float tensor - raw LUT weights (before sigmoid)
            temperature: scalar float

        Returns:
            output: (batch_size, output_dim) float tensor
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError(
                "CUDA extension not available. Please compile probabilistic_cuda extension."
            )

        # Ensure correct dtypes, contiguity, and device placement
        input = input.contiguous().float()
        raw_weights = raw_weights.to(device=input.device, dtype=torch.float32).contiguous()

        # Convert temperature to tensor (C++ binding expects torch.Tensor)
        # Use torch.as_tensor to avoid unnecessary copy warning
        if isinstance(temperature, torch.Tensor):
            temperature_tensor = temperature.to(device=input.device, dtype=torch.float32)
        else:
            temperature_tensor = torch.as_tensor(
                temperature, dtype=torch.float32, device=input.device
            )

        # Call CUDA forward kernel
        output = _probabilistic_cuda_module.forward(input, raw_weights, temperature_tensor)

        # Save for backward
        ctx.save_for_backward(input, raw_weights)
        ctx.temperature = temperature

        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None]:
        """
        Backward pass using CUDA kernel.

        Args:
            grad_output: (batch_size, output_dim) gradient tensor

        Returns:
            Gradients for (input, raw_weights, temperature)
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError(
                "CUDA extension not available. Please compile probabilistic_cuda extension."
            )

        input, raw_weights = ctx.saved_tensors
        temperature = ctx.temperature

        # Ensure contiguity
        grad_output = grad_output.contiguous().float()

        # Convert temperature to tensor (C++ binding expects torch.Tensor)
        # Use torch.as_tensor to avoid unnecessary copy warning
        if isinstance(temperature, torch.Tensor):
            temperature_tensor = temperature.to(device=input.device, dtype=torch.float32)
        else:
            temperature_tensor = torch.as_tensor(
                temperature, dtype=torch.float32, device=input.device
            )

        # Call CUDA backward kernel
        grad_input, grad_weights = _probabilistic_cuda_module.backward(
            input, raw_weights, temperature_tensor, grad_output
        )

        # Return gradients (None for temperature as it doesn't need gradient)
        return grad_input, grad_weights, None


def probabilistic_forward(
    input: torch.Tensor, raw_weights: torch.Tensor, temperature: float
) -> Optional[torch.Tensor]:
    """
    Probabilistic forward pass with automatic differentiation support.

    Args:
        input: (batch_size, input_dim) tensor in [0, 1]
        raw_weights: (output_dim, 2^input_dim) tensor - raw weights before sigmoid
        temperature: scalar float

    Returns:
        output: (batch_size, output_dim) tensor
    """
    if _CUDA_EXT_AVAILABLE and input.is_cuda:
        return ProbabilisticFunction.apply(input, raw_weights, temperature)
    else:
        # CPU fallback handled in forward_train
        return None


@register_node("probabilistic")
class ProbabilisticNode(BaseNode):
    """
    Probabilistic LUT node with continuous inputs in [0,1].
    Uses probabilistic forward pass and autograd for gradients.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        init_fn: Optional[Callable[[torch.Tensor], None]] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        regularizers: Optional[Dict[str, Tuple[Callable, float, Dict[str, Any]]]] = None,
        temperature: float = DEFAULT_PROBABILISTIC_TEMPERATURE,
        eval_mode: str = DEFAULT_PROBABILISTIC_EVAL_MODE,
        use_cuda: bool = DEFAULT_PROBABILISTIC_USE_CUDA,
    ) -> None:
        """
        Initialize a ProbabilisticNode with LUT-based computation.

        Parameters:
        - input_dim: Optional[int], Number of inputs to the node (e.g., 6)
        - output_dim: Optional[int], Number of outputs from the node (e.g., 1)
        - init_fn: Optional[Callable], Initialization function for weights
        - init_kwargs: Optional[Dict[str, Any]], Keyword arguments for init_fn
        - regularizers: Optional[Dict], Custom regularization functions
        - temperature: float, Temperature for sigmoid scaling (default: 1.0)
        - eval_mode: str, Evaluation mode - 'expectation', 'deterministic', or 'threshold'
        - use_cuda: bool, Whether to use CUDA kernels if available (default: True)

        Raises:
            AssertionError: If eval_mode is not one of the valid options
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            regularizers=regularizers,
            init_fn=init_fn,
            init_kwargs=init_kwargs,
        )

        # Warn if temperature was not explicitly provided
        if temperature == DEFAULT_PROBABILISTIC_TEMPERATURE:
            warn_default_value("temperature (probabilistic_node)", temperature, stacklevel=2)

        self.register_buffer("temperature", torch.tensor(float(temperature)))

        # Validate and set eval_mode
        assert eval_mode in {"expectation", "deterministic", "threshold"}, "Invalid eval_mode"
        if eval_mode == DEFAULT_PROBABILISTIC_EVAL_MODE:
            warn_default_value("eval_mode (probabilistic_node)", eval_mode, stacklevel=2)
        self.eval_mode = eval_mode

        # NOTE: use_cuda is no longer stored - CUDA kernels are selected based on device
        # Device determines kernel selection via should_use_cuda_from_tensor()

        # Warn if CUDA extension is available but not provided in init
        if _CUDA_EXT_AVAILABLE and is_cuda_available():
            # Using CUDA implementation
            pass  # Optimal configuration, no warning needed
        elif use_cuda and not _CUDA_EXT_AVAILABLE:
            # User requested CUDA but it's not available
            implementation = "PyTorch GPU (fallback from CUDA)"
            warnings.warn(
                f"ProbabilisticNode: Using {implementation}. "
                f"CUDA extension was requested (use_cuda=True) but is not compiled. "
                f"Compile with: cd difflut && python setup.py install",
                CUDAWarning,
                stacklevel=2,
            )
        elif use_cuda and is_cuda_available():
            # User requested CUDA but couldn't use it (shouldn't happen if _CUDA_EXT_AVAILABLE but doesn't hurt to check)
            implementation = "PyTorch GPU (fallback)"
            pass
        else:
            # use_cuda=False, using CPU
            implementation = "PyTorch CPU"
            warnings.warn(
                f"ProbabilisticNode: Using {implementation}. "
                f"Set use_cuda=True in config and compile CUDA extension for better performance.",
                CUDAWarning,
                stacklevel=2,
            )

        # Store raw weights (logits) - single node instance
        # Shape: (2**input_dim, output_dim)
        self.raw_weights = nn.Parameter(torch.randn(2**self.input_dim, self.output_dim))
        self._apply_init_fn(self.raw_weights, name="raw_weights")

        # Precompute all binary combinations (LSB-first order) - for CPU fallback
        binary_combinations = []
        for i in range(2**self.input_dim):
            bits = [((i >> j) & 1) for j in range(self.input_dim)]  # LSB first
            binary_combinations.append(bits)
        self.register_buffer(
            "binary_combinations", torch.tensor(binary_combinations, dtype=torch.float32)
        )

    @property
    def weights(self) -> torch.Tensor:
        """Get weights with sigmoid applied (temperature scaled) in [0,1]."""
        # Ensure temperature is on same device as raw_weights
        temperature = self.temperature.to(device=self.raw_weights.device)
        return torch.sigmoid(self.raw_weights / temperature.clamp(min=1e-6))

    def _binary_to_index(self, x_binary: torch.Tensor) -> torch.Tensor:
        """Convert binary vector to LUT index (LSB-first order)"""
        powers = 2 ** torch.arange(self.input_dim, device=x_binary.device, dtype=torch.float32)
        if x_binary.dim() == 1:
            return (x_binary * powers).sum().long()
        else:
            return (x_binary * powers).sum(dim=-1).long()

    def _bernoulli_probability(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Compute Pr(a|x) = Π_j [x_j^a_j * (1-x_j)^(1-a_j)]
        Args:
            x: continuous inputs in [0,1], shape (batch_size, num_inputs)
            a: binary pattern, shape (num_inputs,) or (batch_size, num_inputs)
        """
        if a.dim() == 1:
            a = a.unsqueeze(0).expand(x.shape[0], -1)

        # Compute probability
        prob = x * a + (1 - x) * (1 - a)
        return torch.prod(prob, dim=1)

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training: probabilistic expectation (vectorized)
        f(x) = Σ_a ω_δ(a) * Pr(a|x)
        Uses CUDA kernel if available, otherwise falls back to CPU.

        Args:
            x: Input tensor (batch_size, input_dim)
        Returns:
            Output tensor (batch_size, output_dim)
        """
        batch_size, input_dim = x.shape

        # Try CUDA kernel first based on tensor device
        # Device determines kernel selection, not config parameters
        # BOTH input and weights must be on CUDA for the CUDA kernel
        if (
            should_use_cuda_from_tensor(x)
            and should_use_cuda_from_tensor(self.raw_weights)
            and _CUDA_EXT_AVAILABLE
        ):
            # raw_weights shape: (2^input_dim, output_dim)
            # Ensure temperature is on same device as raw_weights
            temperature_value = self.temperature.to(device=self.raw_weights.device).item()
            output = probabilistic_forward(x, self.raw_weights, temperature_value)
            if output is not None:
                return output

        # CPU fallback
        # weights shape: (2^input_dim, output_dim)
        weights = self.weights

        # Ensure binary_combinations is on the same device and dtype as x
        binary_combinations = self.binary_combinations.to(device=x.device, dtype=x.dtype)

        # Vectorized probability computation
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        a_expanded = binary_combinations.unsqueeze(0)  # (1, 2^input_dim, input_dim)
        prob_terms = x_expanded * a_expanded + (1 - x_expanded) * (1 - a_expanded)
        probs = torch.prod(prob_terms, dim=-1)  # (batch_size, 2^input_dim)

        # Apply weights: (batch_size, 2^input_dim) @ (2^input_dim, output_dim) -> (batch_size, output_dim)
        output = torch.matmul(probs, weights)

        return output

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Direct LUT lookup simulating hardware behavior.
        Assumes inputs are already binary (0 or 1) from encoder or previous nodes.
        Returns binary outputs (0 or 1).

        Args:
            x: Input tensor (batch_size, input_dim)
        Returns:
            Output tensor (batch_size, output_dim)
        """
        batch_size, input_dim = x.shape

        # Convert binary inputs to LUT indices (LSB-first order)
        powers = 2 ** torch.arange(input_dim, device=x.device, dtype=x.dtype)
        indices = (x * powers).sum(dim=-1).long()  # (batch_size,)

        # Look up weights and threshold at 0.5 to get binary output
        weights = self.weights  # (2^input_dim, output_dim)

        # Ensure indices are on same device as weights
        indices = indices.to(device=weights.device)

        # Gather weights: (batch_size, output_dim)
        output = weights[indices]  # (batch_size, output_dim)

        # Threshold to get binary output (weights are in [0, 1] after sigmoid)
        output = (output >= 0.5).float()

        return output

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization by default."""
        return torch.tensor(0.0, device=self.raw_weights.device)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"
