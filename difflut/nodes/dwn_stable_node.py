import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..registry import register_node
from ..utils.warnings import CUDAWarning, warn_default_value
from .base_node import BaseNode
from .cuda import is_cuda_available

# Default gradient scaling factor for DWN Stable nodes
DEFAULT_DWN_STABLE_GRADIENT_SCALE: float = 1.0
# Default flag for using CUDA kernels in DWN Stable nodes
DEFAULT_DWN_STABLE_USE_CUDA: bool = True

# Try to import the compiled CUDA extension
try:
    import dwn_stable_cuda as _dwn_stable_cuda_module

    _CUDA_EXT_AVAILABLE = True
except ImportError:
    _CUDA_EXT_AVAILABLE = False
    _dwn_stable_cuda_module = None
    warnings.warn(
        "CUDA extension 'dwn_stable_cuda' not available. DWNStableNode will use slower CPU fallback. "
        "For better performance, compile the CUDA extension using: "
        "'cd difflut && python setup.py install'. "
        "To suppress this warning: warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut.nodes.dwn_stable_node')",
        RuntimeWarning,
        stacklevel=2,
    )


class GradientStabilizedFunction(torch.autograd.Function):
    """
    PyTorch autograd function wrapper for Gradient Stabilized CUDA kernels.
    Processes 2D tensors with gradient scaling in backward pass.
    Same as EFD but with gradient scaling in backward pass.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        luts: torch.Tensor,
        gradient_scale: float,
    ) -> torch.Tensor:
        """
        Forward pass using CUDA kernel.

        Parameters:
        - input: torch.Tensor, (batch_size, input_dim) float tensor in [0, 1]
        - luts: torch.Tensor, (output_dim, 2^input_dim) float tensor in [0, 1]
        - gradient_scale: float, scalar float for gradient scaling
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError(
                "CUDA extension not available. Please compile dwn_stable_cuda extension."
            )

        # Ensure correct dtypes and contiguity
        input = input.contiguous().float()
        luts = luts.contiguous().float()

        # Call CUDA forward kernel
        output = _dwn_stable_cuda_module.forward(input, luts)

        # Save for backward
        ctx.save_for_backward(input, luts)
        ctx.gradient_scale = gradient_scale

        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """
        Backward pass using CUDA kernel with gradient scaling.

        Parameters:
        - grad_output: torch.Tensor, (batch_size, output_dim) gradient tensor
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError(
                "CUDA extension not available. Please compile dwn_stable_cuda extension."
            )

        input, luts = ctx.saved_tensors
        gradient_scale = ctx.gradient_scale

        # Ensure contiguity
        grad_output = grad_output.contiguous().float()

        # Call CUDA backward kernel with gradient scaling
        grad_input, grad_luts = _dwn_stable_cuda_module.backward(
            input, luts, grad_output, gradient_scale
        )

        # Return gradients (None for gradient_scale)
        return grad_input, grad_luts, None


class GradientStabilizedFunctionCPU(torch.autograd.Function):
    """
    CPU fallback for Gradient Stabilized with EFD backward pass and gradient scaling.
    Processes 2D tensors.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        luts: torch.Tensor,
        gradient_scale: float,
    ) -> torch.Tensor:
        """Forward pass with binary thresholding."""
        # Binary threshold at 0.5 for [0, 1] inputs
        x_binary = (input >= 0.5).float()
        batch_size, input_dim = x_binary.shape
        output_dim = luts.shape[0]

        # Compute LUT indices from binary inputs
        powers = 2 ** torch.arange(input_dim, device=input.device, dtype=torch.float32)
        indices = (x_binary * powers).sum(dim=-1).long()  # (batch_size,)

        # Look up LUT values: luts is (output_dim, 2^input_dim)
        # indices is (batch_size,)
        # We want output[b, o] = luts[o, indices[b]]
        output = luts[:, indices].T  # (batch_size, output_dim)

        # Save for backward
        ctx.save_for_backward(input, luts)
        ctx.gradient_scale = gradient_scale

        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """Backward pass using Gradient Stabilized EFD with gradient scaling."""
        input, luts = ctx.saved_tensors
        gradient_scale = ctx.gradient_scale

        batch_size, input_dim = input.shape
        output_dim = luts.shape[0]
        lut_size = 2**input_dim

        grad_input = torch.zeros_like(input)
        grad_luts = torch.zeros_like(luts)

        for batch_idx in range(batch_size):
            # Compute current address from binary input
            addr = 0
            for i in range(input_dim):
                if input[batch_idx, i].item() >= 0.5:
                    addr |= 1 << i

            # LUT gradient - direct assignment to accessed entry
            for dim_idx in range(output_dim):
                grad_luts[dim_idx, addr] += (
                    grad_output[batch_idx, dim_idx] * gradient_scale
                )

            # Input gradient using Gradient Stabilized EFD
            for input_idx in range(input_dim):
                # Create mask to exclude input_idx-th bit
                mask = ((1 << input_dim) - 1) & ~(1 << input_idx)
                addr_masked = addr & mask

                total_gradient = 0.0

                # Iterate over all possible addresses k
                for k in range(lut_size):
                    # Calculate Hamming distance between addr and k, excluding input_idx-th bit
                    k_masked = k & mask
                    hamming_dist = bin(addr_masked ^ k_masked).count("1")

                    # Get k_l (input_idx-th bit of k)
                    k_l = (k >> input_idx) & 1

                    # Calculate sign factor: (-1)^(1-k_l)
                    sign_factor = -1.0 if k_l == 0 else 1.0

                    # Get LUT values at position k for all output dimensions
                    for dim_idx in range(output_dim):
                        lut_value = luts[dim_idx, k].item()

                        # Gradient stabilized formula: sign * lut / (hamming_dist + 1)
                        total_gradient += (
                            sign_factor
                            * lut_value
                            / (hamming_dist + 1.0)
                            * gradient_scale
                            * grad_output[batch_idx, dim_idx].item()
                        )

                grad_input[batch_idx, input_idx] = total_gradient

        return grad_input, grad_luts, None


def dwn_stable_forward(
    input: torch.Tensor, luts: torch.Tensor, gradient_scale: float
) -> Optional[torch.Tensor]:
    """
    Gradient Stabilized forward pass with automatic differentiation support.

    Parameters:
    - input: torch.Tensor, (batch_size, input_dim) tensor in [0, 1]
    - luts: torch.Tensor, (output_dim, 2^input_dim) tensor in [0, 1]
    - gradient_scale: float, scalar float for gradient scaling
    """
    if _CUDA_EXT_AVAILABLE and input.is_cuda:
        return GradientStabilizedFunction.apply(input, luts, gradient_scale)
    else:
        # CPU fallback with proper gradient stabilized EFD backward
        return GradientStabilizedFunctionCPU.apply(input, luts, gradient_scale)


@register_node("dwn_stable")
class DWNStableNode(BaseNode):
    """
    Gradient Stabilized Node - Same as DWN but with gradient scaling.

    Forward: Binary thresholding at 0.5 (>= 0.5) for inputs in [0, 1]
    Backward: Extended Finite Difference (EFD) multiplied by gradient_scale

    Weights: Raw weights passed through sigmoid to get LUT values in [0, 1]
    Processes 2D tensors: (batch_size, input_dim) -> (batch_size, output_dim)
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        use_cuda: bool = True,
        regularizers: Optional[
            Dict[str, Tuple[Callable, float, Dict[str, Any]]]
        ] = None,
        gradient_scale: float = 1.25,
        init_fn: Optional[Callable] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Gradient Stabilized Node - Same as DWN but with gradient scaling.

        Parameters:
        - input_dim: Optional[int], Number of inputs (e.g., 6), (default: None)
        - output_dim: Optional[int], Number of outputs (e.g., 1), (default: None)
        - use_cuda: bool, Whether to use CUDA kernels (if available), (default: True)
        - regularizers: Optional[Dict], Custom regularization functions, (default: None)
        - gradient_scale: float, Initial gradient scaling factor (learnable), (default: 1.25)
        - init_fn: Optional[Callable], Initialization function for LUT weights, (default: None)
        - init_kwargs: Optional[Dict[str, Any]], Keyword arguments for init_fn, (default: None)
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            regularizers=regularizers,
            init_fn=init_fn,
            init_kwargs=init_kwargs,
        )
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
                f"DWNStableNode: Using {implementation}. "
                f"CUDA extension was requested (use_cuda=True) but is not compiled. "
                f"Compile with: cd difflut && python setup.py install",
                CUDAWarning,
                stacklevel=2,
            )
        else:
            # use_cuda=False, using CPU
            implementation = "PyTorch CPU"
            warnings.warn(
                f"DWNStableNode: Using {implementation}. "
                f"Set use_cuda=True in config and compile CUDA extension for better performance.",
                CUDAWarning,
                stacklevel=2,
            )

        # Learnable gradient scaling factor
        self.gradient_scale = nn.Parameter(torch.tensor(gradient_scale))

        # Initialize raw LUT weights
        # Shape: (output_dim, 2^input_dim)
        lut_size = 2**self.num_inputs
        self.raw_luts = nn.Parameter(torch.randn(self.num_outputs, lut_size) * 0.1)

        # Apply initialization to raw_luts if init_fn is provided
        self._apply_init_fn(self.raw_luts, name="raw_luts")

    def _get_luts(self) -> torch.Tensor:
        """
        Get actual LUT weights by applying sigmoid to raw weights.
        Maps from (-inf, inf) to [0, 1].
        Returns: (output_dim, 2^input_dim)
        """
        return torch.sigmoid(self.raw_luts)

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training.
        Inputs are in [0, 1], binarized using Heaviside at 0.5.
        Outputs are in [0, 1].
        Uses CUDA kernels when available, otherwise CPU fallback.

        Args:
            x: Input tensor (batch_size, input_dim)
        Returns:
            Output tensor (batch_size, output_dim)
        """
        # Ensure input is on the same device as parameters
        x = x.to(self.raw_luts.device)

        # Get actual LUT weights via sigmoid: (output_dim, 2^input_dim)
        luts = self._get_luts()

        # Use CUDA if available based on tensor device
        # Device determines kernel selection, not config parameters
        # BOTH input and weights must be on CUDA for the CUDA kernel
        if x.is_cuda and self.raw_luts.is_cuda and _CUDA_EXT_AVAILABLE:
            output = dwn_stable_forward(x, luts, self.gradient_scale.item())
        else:
            # CPU fallback
            output = dwn_stable_forward(x, luts, self.gradient_scale.item())

        return output

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Inputs already binarized in {0, 1}.
        Output binarized to {0, 1} using Heaviside at 0.5.

        Args:
            x: Input tensor (batch_size, input_dim)
        Returns:
            Output tensor (batch_size, output_dim)
        """
        # Get actual LUT weights via sigmoid: (output_dim, 2^input_dim)
        luts = self._get_luts()

        # Inputs are already binarized in {0, 1}, use directly
        x_binary = x.float()

        # Compute LUT indices from binary inputs
        powers = 2 ** torch.arange(
            self.num_inputs, device=x.device, dtype=torch.float32
        )
        indices = (x_binary * powers).sum(dim=-1).long()  # (batch_size,)

        # Look up LUT values: luts is (output_dim, lut_size)
        # indices is (batch_size,)
        # We want output[b, o] = luts[o, indices[b]]
        output = luts[:, indices].T  # (batch_size, output_dim)

        # Binarize output: [0, 1] -> {0, 1} using Heaviside at 0.5
        output = (output >= 0.5).float()

        return output

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization to match base CUDA implementation."""
        return torch.tensor(0.0, device=self.raw_luts.device, requires_grad=False)
