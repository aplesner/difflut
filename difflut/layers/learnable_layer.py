import warnings
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nodes.node_config import NodeConfig
from ..registry import register_layer
from ..utils.warnings import warn_default_value
from .base_layer import BaseLUTLayer
from .layer_config import LayerConfig

# Default temperature for softmax in learnable mapping
DEFAULT_LEARNABLE_LAYER_TAU: float = 0.001
# Default starting value for tau (used for exponential decay)
DEFAULT_LEARNABLE_LAYER_TAU_START: float = 1.0
# Default minimum value tau can decay to
DEFAULT_LEARNABLE_LAYER_TAU_MIN: float = 0.0001
# Default number of iterations for tau to decay by factor of 10
DEFAULT_LEARNABLE_LAYER_TAU_DECAY_ITERS: float = 1000.0
# Threshold for warning about excessive learnable connections
# If output_size * n > input_size * LEARNABLE_LAYER_CONNECTION_WARNING_THRESHOLD, warn
LEARNABLE_LAYER_CONNECTION_WARNING_THRESHOLD: int = 10
# Use CUDA kernel for soft selection (training mode)
# If False, always use PyTorch's matmul (more stable, well-tested)
# If True, use custom CUDA kernel (potentially faster for large matrices)
# Note: Hard selection (eval mode) always uses CUDA kernel when available
DEFAULT_LEARNABLE_LAYER_USE_CUDA_SOFT: bool = False

# Try to import the compiled CUDA extension for learnable mapping
try:
    # pyright: ignore[reportMissingImports]
    import learnable_mapping_cuda as _learnable_mapping_cuda_module

    _LEARNABLE_MAPPING_CUDA_AVAILABLE = True
except ImportError:
    _LEARNABLE_MAPPING_CUDA_AVAILABLE = False
    _learnable_mapping_cuda_module = None
    warnings.warn(
        "CUDA extension 'learnable_mapping_cuda' not available. LearnableLayer will use PyTorch fallback. "
        "For better performance, compile the CUDA extension using: "
        "'cd difflut && python setup.py install'. "
        "To suppress this warning: warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut.layers.learnable_layer')",
        RuntimeWarning,
        stacklevel=2,
    )

# Try to import the compiled CUDA extension for probabilistic nodes
try:
    # pyright: ignore[reportMissingImports]
    import probabilistic_cuda as _probabilistic_cuda_module

    _PROBABILISTIC_CUDA_AVAILABLE = True
except ImportError:
    _PROBABILISTIC_CUDA_AVAILABLE = False
    _probabilistic_cuda_module = None


class LearnableMappingFunction(torch.autograd.Function):
    """
    Autograd function wrapper for learnable mapping CUDA kernel (hard selection).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        indices: torch.Tensor,
        input_size: int,
    ) -> torch.Tensor:
        """
        Forward pass using CUDA kernel for hard selection.

        Parameters:
        - input: torch.Tensor, (batch_size, input_size) float tensor
        - indices: torch.Tensor, (output_size,) int32 tensor - argmax results
        - input_size: int, needed for backward
        """
        if not _LEARNABLE_MAPPING_CUDA_AVAILABLE or _learnable_mapping_cuda_module is None:
            raise RuntimeError("CUDA extension not available. Use fallback implementation.")

        # Ensure correct dtypes and contiguity
        input = input.contiguous().float()
        indices = indices.contiguous().int()

        # Call CUDA forward kernel
        output = _learnable_mapping_cuda_module.forward(input, indices)

        # Save for backward
        ctx.save_for_backward(indices)
        # pyright: ignore[reportAttributeAccessIssue]
        ctx.input_size = input_size

        return output

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None]:
        """
        Backward pass using CUDA kernel.

        Parameters:
        - grad_output: torch.Tensor, (batch_size, output_size) gradient tensor
        """
        if not _LEARNABLE_MAPPING_CUDA_AVAILABLE or _learnable_mapping_cuda_module is None:
            raise RuntimeError("CUDA extension not available.")

        # pyright: ignore[reportAttributeAccessIssue]
        (indices,) = ctx.saved_tensors
        # pyright: ignore[reportAttributeAccessIssue]
        input_size = ctx.input_size

        # Ensure contiguity
        grad_output = grad_output.contiguous().float()

        # Call CUDA backward kernel
        grad_input = _learnable_mapping_cuda_module.backward(grad_output, indices, input_size)

        # Return gradients (None for indices and input_size)
        return grad_input, None, None


def learnable_mapping_forward_cuda(
    input: torch.Tensor, indices: torch.Tensor, input_size: int
) -> Optional[torch.Tensor]:
    """
    Learnable mapping forward pass (hard selection) with CUDA.

    Parameters:
    - input: torch.Tensor, (batch_size, input_size) tensor
    - indices: torch.Tensor, (output_size,) tensor
    - input_size: int, size of input
    """
    if _LEARNABLE_MAPPING_CUDA_AVAILABLE and input.is_cuda:
        return LearnableMappingFunction.apply(input, indices, input_size)
    else:
        return None


def learnable_mapping_soft_forward_cuda(
    input: torch.Tensor, weights: torch.Tensor, tau: float
) -> Optional[torch.Tensor]:
    """
    Learnable mapping forward pass (soft selection) with CUDA.

    Parameters:
    - input: torch.Tensor, (batch_size, input_size) tensor
    - weights: torch.Tensor, (output_size, input_size) tensor
    - tau: float, temperature parameter
    """
    if (
        _LEARNABLE_MAPPING_CUDA_AVAILABLE
        and input.is_cuda
        and _learnable_mapping_cuda_module is not None
    ):
        try:
            return _learnable_mapping_cuda_module.soft_forward(input, weights, tau)
        except:
            return None
    else:
        return None


class LearnableMappingModule(nn.Module):
    """
    Helper module for learnable mapping (not registered, used internally).
    Provides soft selection during training and hard selection during evaluation.
    Uses CUDA kernels when available for optimal performance.

    Note: output_size here is actually (layer_output_size * n) - the total number of
    selections needed. This module doesn't know about the layer structure.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tau: float = DEFAULT_LEARNABLE_LAYER_TAU,
        use_cuda_soft: bool = DEFAULT_LEARNABLE_LAYER_USE_CUDA_SOFT,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size  # This is actually layer_output_size * n
        self.tau = tau
        # NOTE: use_cuda_soft is no longer stored - CUDA kernels are selected based on device
        # Device determines kernel selection via should_use_cuda_from_tensor()

        # Weight matrix: (output_size, input_size) where output_size = layer_output_size * n
        self.W = nn.Parameter(torch.randn(output_size, input_size))
        nn.init.xavier_uniform_(self.W)

        # Cache for hard selection (indices instead of mask for CUDA kernel)
        self.register_buffer("_cached_hard_indices", None)
        # Keep for PyTorch fallback
        self.register_buffer("_cached_hard_mask", None)
        self._cache_valid = False

    def train(self, mode: bool = True):
        """Override train() to invalidate cache when switching modes."""
        was_training = self.training
        super().train(mode)

        # Invalidate cache when switching from eval to train
        if mode and not was_training:
            self._cache_valid = False

        return self

    def _compute_hard_selection(self) -> None:
        """
        Compute hard selection from current weights.
        Computes both indices (for CUDA kernel) and mask (for PyTorch fallback).
        """
        with torch.no_grad():
            # Get hard indices: which input is selected for each output position
            # W shape: (output_size, input_size) where output_size = layer_output_size * n
            hard_indices = torch.argmax(self.W, dim=-1)  # (output_size,)

            # Store indices for CUDA kernel
            self._cached_hard_indices = hard_indices.int()

            # Also create binary mask for PyTorch fallback
            # mask: (input_size, output_size) where mask[i, o] = 1 if input i is selected for output o
            mask = torch.zeros(
                (self.input_size, self.output_size), dtype=torch.uint8, device=self.W.device
            )
            output_indices = torch.arange(self.output_size, device=self.W.device)
            mask[hard_indices, output_indices] = 1
            self._cached_hard_mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Soft selection (training) or hard selection (eval).
        Uses CUDA kernels when available based on tensor device.
        Training: softmax + matmul (PyTorch default, or optional CUDA kernel)
        Eval: CUDA kernel for direct lookup (faster than einsum)
        """
        # Ensure input is on the same device as parameters
        x = x.to(self.W.device)
        
        if self.training:
            # Soft selection - training mode
            # Try CUDA kernel first if available based on tensor device
            # Device determines kernel selection, not config parameters
            if x.is_cuda and _LEARNABLE_MAPPING_CUDA_AVAILABLE:
                output = learnable_mapping_soft_forward_cuda(x, self.W, self.tau)
                if output is not None:
                    return output

            # PyTorch fallback - already well optimized
            weights = F.softmax(self.W / self.tau, dim=-1)
            output = torch.matmul(x, weights.t())
        else:
            # Hard selection (evaluation mode)
            # OPTIMIZATION: Cache hard selection to avoid repeated argmax computation
            if not self._cache_valid or self._cached_hard_indices is None:
                self._compute_hard_selection()
                self._cache_valid = True

            # Try CUDA kernel first based on tensor device (fastest)
            # Device determines kernel selection, not config parameters
            if x.is_cuda and _LEARNABLE_MAPPING_CUDA_AVAILABLE:
                output = learnable_mapping_forward_cuda(
                    x, self._cached_hard_indices, self.input_size
                )
                if output is not None:
                    return output

            # PyTorch fallback - use einsum with binary mask
            # x: (batch_size, input_size) -> (b, i)
            # _cached_hard_mask: (input_size, output_size) -> (i, o)
            # Result: (batch_size, output_size) -> (b, o)
            mask_float = self._cached_hard_mask.float()
            output = torch.einsum("bi,io->bo", x, mask_float)

        return output


@register_layer("learnable")
class LearnableLayer(BaseLUTLayer):
    """
    LUT layer with learnable mapping using nodes.
    Uses soft selection during training and hard selection during evaluation.

    The learnable mapping uses a weight matrix W and applies softmax for soft
    selection during training, or argmax for hard selection during evaluation.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        node_type: Type[nn.Module],
        node_kwargs: NodeConfig,
        tau: Optional[float] = None,
        tau_start: Optional[float] = None,
        tau_min: Optional[float] = None,
        tau_decay_iters: Optional[float] = None,
        layer_config: Optional[LayerConfig] = None,
        flip_probability: Optional[float] = None,
        grad_stabilization: Optional[str] = None,
        grad_target_std: Optional[float] = None,
        grad_subtract_mean: Optional[bool] = None,
        grad_epsilon: Optional[float] = None,
        use_cuda_soft: Optional[bool] = None,
    ):
        """
        Learnable Layer with learnable input mapping.

        Parameters:
        - input_size: int, Size of input vector (from encoder or previous layer)
        - output_size: int, Number of LUT nodes
        - node_type: Type, LUT node class
        - node_kwargs: NodeConfig, Node configuration (input_dim, output_dim, etc.)
        - tau: Optional[float], Initial temperature for softmax in learnable mapping, (default: None)
        - tau_start: Optional[float], Starting value for tau (exponential decay), (default: None)
        - tau_min: Optional[float], Minimum value tau can decay to, (default: None)
        - tau_decay_iters: Optional[float], Number of iterations for tau decay, (default: None)
        - layer_config: Optional[LayerConfig], Training parameters (flip_probability, grad_stabilization, etc.), (default: None)
        - flip_probability: Optional[float], Probability of flipping each bit during training, (default: None)
        - grad_stabilization: Optional[str], Gradient stabilization mode ('none', 'layerwise', 'batchwise'), (default: None)
        - grad_target_std: Optional[float], Target standard deviation for gradient rescaling, (default: None)
        - grad_subtract_mean: Optional[bool], Whether to subtract mean before rescaling, (default: None)
        - grad_epsilon: Optional[float], Small constant for numerical stability, (default: None)
        - use_cuda_soft: Optional[bool], Use CUDA kernel for soft selection (training mode), (default: None)
        """
        # Set defaults from constants
        if tau is None:
            tau = DEFAULT_LEARNABLE_LAYER_TAU
            warn_default_value("tau", tau, stacklevel=2)
        if tau_start is None:
            tau_start = DEFAULT_LEARNABLE_LAYER_TAU_START
            warn_default_value("tau_start", tau_start, stacklevel=2)
        if tau_min is None:
            tau_min = DEFAULT_LEARNABLE_LAYER_TAU_MIN
            warn_default_value("tau_min", tau_min, stacklevel=2)
        if tau_decay_iters is None:
            tau_decay_iters = DEFAULT_LEARNABLE_LAYER_TAU_DECAY_ITERS
            warn_default_value("tau_decay_iters", tau_decay_iters, stacklevel=2)
        if use_cuda_soft is None:
            use_cuda_soft = DEFAULT_LEARNABLE_LAYER_USE_CUDA_SOFT
            warn_default_value("use_cuda_soft", use_cuda_soft, stacklevel=2)

        # Warn if tau parameters seem unusual
        if tau_start < tau_min:
            warnings.warn(
                f"LearnableLayer: tau_start ({tau_start}) is less than tau_min ({tau_min}). "
                f"This means tau will be clamped immediately. Set tau_start >= tau_min.",
                UserWarning,
                stacklevel=2,
            )

        # Initialize parent with nodes (n will be extracted from created nodes)
        super().__init__(
            input_size,
            output_size,
            node_type,
            node_kwargs,
            layer_config,
            flip_probability,
            grad_stabilization,
            grad_target_std,
            grad_subtract_mean,
            grad_epsilon,
        )

        # Warn about parameter count after n is known
        total_connections = output_size * self.n
        if total_connections > input_size * LEARNABLE_LAYER_CONNECTION_WARNING_THRESHOLD:
            warnings.warn(
                f"LearnableLayer: Creating {total_connections} learnable connections from {input_size} inputs. "
                f"This may lead to overfitting. Consider using GroupedLayer or fewer nodes/inputs per node (n={self.n}).",
                UserWarning,
                stacklevel=2,
            )

        # Tau decay parameters
        self.tau_start = tau_start
        self.tau_min = tau_min
        self.tau_decay_iters = tau_decay_iters
        self.tau = tau_start  # Start with tau_start instead of tau
        # NOTE: use_cuda_soft is no longer stored - CUDA kernels are selected based on device
        # Device determines kernel selection via should_use_cuda_from_tensor()

        # Create learnable mapping module (helper, not registered)
        # Note: self.n is now available from parent's __init__
        self.mapping = LearnableMappingModule(
            input_size, output_size * self.n, self.tau, use_cuda_soft
        )

    def get_mapping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable mapping and reshape for nodes.

        Uses efficient matrix operations (training) or advanced indexing (eval).

        Args:
            x: Input tensor of shape (batch_size, input_size)
        Returns:
            Mapped inputs of shape (batch_size, output_size, n)
        """
        batch_size = x.shape[0]

        # Apply learnable mapping (already optimized in LearnableMappingModule)
        # Training: uses matmul (efficient)
        # Eval: uses advanced indexing (efficient)
        mapped_flat = self.mapping(x)  # (batch_size, output_size * n)

        # Reshape for nodes
        mapped_inputs = mapped_flat.view(batch_size, self.output_size, self.n)

        return mapped_inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through learnable mapping and node.

        Parameters:
        - x: torch.Tensor, Input tensor of shape (batch_size, input_size)
        """
        # Validate input dimensions
        self._validate_input_dims(x)

        # Get mapped inputs: (batch_size, output_size, n)
        mapped_inputs = self.get_mapping(x)

        batch_size = mapped_inputs.shape[0]
        # type: ignore[reportAttributeAccessIssue]
        output_dim: int = self.nodes[0].output_dim

        # Preallocate output tensor
        output = torch.empty(
            (batch_size, self.output_size, output_dim), device=x.device, dtype=x.dtype
        )

        # Process each node independently with its slice of mapped inputs
        for node_idx, node in enumerate(self.nodes):
            # Extract inputs for this node: (batch_size, n)
            node_input = mapped_inputs[:, node_idx, :]
            # Forward through node: (batch_size, n) -> (batch_size, output_dim)
            output[:, node_idx, :] = node(node_input)

        # Reshape to 2D for next layer: (batch_size, output_size * output_dim)
        output = output.view(batch_size, -1)

        return output

    def get_mapping_matrix(self) -> torch.Tensor:
        """Get current hard mapping (for inspection) as indices."""
        with torch.no_grad():
            # Get hard indices from weight matrix
            hard_indices = torch.argmax(self.mapping.W, dim=-1)  # (output_size * n,)
            return hard_indices.view(self.output_size, self.n)

    def update_tau(self, iteration: int):
        """
        Update tau using exponential decay.

        Args:
            iteration: Current training iteration
        """
        # Calculate decay factor: tau = tau_start * 10^(-iteration / tau_decay_iters)
        # This means tau decays by a factor of 10 every tau_decay_iters iterations
        decay_factor = 10.0 ** (-iteration / self.tau_decay_iters)
        self.tau = max(self.tau_start * decay_factor, self.tau_min)

        # Update the mapping module's tau
        self.mapping.tau = self.tau

        # Invalidate cache since tau change might affect hard selection
        # (though in practice, argmax is deterministic regardless of tau)
        self.mapping._cache_valid = False

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        flip_str = f", flip_prob={self.flip_probability}" if self.flip_probability > 0 else ""
        grad_str = (
            f", grad_stab={self.grad_stabilization}" if self.grad_stabilization != "none" else ""
        )
        return (
            f"input_size={self.input_size}, output_size={self.output_size}, "
            f"n={self.n}, tau={self.tau}, mapping=learnable{flip_str}{grad_str}"
        )
