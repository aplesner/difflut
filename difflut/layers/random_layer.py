import warnings
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn

from ..nodes.node_config import NodeConfig
from ..registry import register_layer
from .base_layer import BaseLUTLayer
from .layer_config import LayerConfig

# Default random seed for reproducible random mapping
DEFAULT_RANDOM_LAYER_SEED: int = 42


# Try to import the compiled CUDA extension for mapping
try:
    import mapping_cuda as _mapping_cuda_module  # pyright: ignore[reportMissingImports]

    _MAPPING_CUDA_AVAILABLE = True
except ImportError:
    _MAPPING_CUDA_AVAILABLE = False
    _mapping_cuda_module = None
    warnings.warn(
        "CUDA extension 'mapping_cuda' not available. RandomLayer will use slower PyTorch fallback. "
        "For better performance, compile the CUDA extension using: "
        "'cd difflut && python setup.py install'. "
        "To suppress this warning: warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut.layers.random_layer')",
        RuntimeWarning,
        stacklevel=2,
    )


class MappingFunction(torch.autograd.Function):
    """
    Autograd function wrapper for mapping CUDA kernel.
    Provides forward and backward passes with automatic differentiation.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        indices: torch.Tensor,
        input_size: int,
    ) -> torch.Tensor:
        """
        Forward pass using CUDA kernel.

        Args:
            input: (batch_size, input_size) float tensor
            indices: (output_size, n) int16/int32 tensor
            input_size: int (needed for backward)

        Returns:
            output: (batch_size, output_size, n) float tensor
        """
        if not _MAPPING_CUDA_AVAILABLE or _mapping_cuda_module is None:
            raise RuntimeError("CUDA extension not available. Use fallback implementation.")

        # Ensure correct dtypes and contiguity
        input = input.contiguous().float()
        indices = indices.contiguous()

        # Call CUDA forward kernel
        output = _mapping_cuda_module.forward(input, indices)

        # Save for backward
        ctx.save_for_backward(indices)
        ctx.input_size = input_size  # pyright: ignore[reportAttributeAccessIssue]

        return output

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None]:
        """
        Backward pass using CUDA kernel.

        Args:
            grad_output: (batch_size, output_size, n) gradient tensor

        Returns:
            Gradients for (input, indices, input_size)
        """
        if not _MAPPING_CUDA_AVAILABLE or _mapping_cuda_module is None:
            raise RuntimeError("CUDA extension not available.")

        (indices,) = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
        input_size = ctx.input_size  # pyright: ignore[reportAttributeAccessIssue]

        # Ensure contiguity
        grad_output = grad_output.contiguous().float()

        # Call CUDA backward kernel
        grad_input = _mapping_cuda_module.backward(grad_output, indices, input_size)

        # Return gradients (None for indices and input_size as they don't need gradients)
        return grad_input, None, None


def mapping_forward_cuda(
    input: torch.Tensor, indices: torch.Tensor, input_size: int
) -> Optional[torch.Tensor]:
    """
    Mapping forward pass with automatic differentiation support.

    Args:
        input: (batch_size, input_size) tensor
        indices: (output_size, n) tensor
        input_size: int

    Returns:
        output: (batch_size, output_size, n) tensor
    """
    if _MAPPING_CUDA_AVAILABLE and input.is_cuda:
        return MappingFunction.apply(input, indices, input_size)
    else:
        return None


@register_layer("random")
class RandomLayer(BaseLUTLayer):
    """
    LUT layer with purely random fixed mapping.
    Each input is used at least once per node before being reused.
    Connections are randomly initialized and remain fixed during training.

    Uses efficient index_select operations to minimize memory during mapping,
    avoiding large intermediate tensors.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        node_type: Type[nn.Module],
        node_kwargs: NodeConfig,
        seed: Optional[int] = DEFAULT_RANDOM_LAYER_SEED,
        layer_config: Optional[LayerConfig] = None,
        flip_probability: Optional[float] = None,
        grad_stabilization: Optional[str] = None,
        grad_target_std: Optional[float] = None,
        grad_subtract_mean: Optional[bool] = None,
        grad_epsilon: Optional[float] = None,
        ensure_full_input_coverage: bool = True,
        mapping_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            input_size: Size of input vector (from encoder or previous layer)
                       Should match: (batch_size, input_size)
            output_size: Number of LUT nodes (output will be batch_size, output_size * output_dim)
            node_type: LUT node class to use
            node_kwargs: Node configuration (NodeConfig instance with input_dim, output_dim, etc.)
                        Dimension spec: nodes expect (batch_size, output_size, node_input_dim)
            seed: Random seed for reproducible mapping
            layer_config: LayerConfig object with training parameters (flip_probability, grad_stabilization, etc.)
            flip_probability: Probability of flipping each bit during training (0.0 to 1.0)
            grad_stabilization: Gradient stabilization mode ('none', 'layerwise', 'batchwise')
            grad_target_std: Target standard deviation for gradient rescaling
            grad_subtract_mean: Whether to subtract mean before rescaling
            grad_epsilon: Small constant for numerical stability
            ensure_full_input_coverage: If True, ensures each input is used at least once per node
                                        before any input is reused. If False, inputs are sampled independently.
            mapping_indices: Optional pre-defined mapping indices tensor of shape (output_size, n).
                             If provided, this mapping will be used instead of generating a new (random) one.
        """
        self.seed = seed

        # Note: Seed warning removed since seed is now explicitly provided in configs.
        # Only warn if seed is truly missing (None), not if it equals the default value.

        # Initialize parent (n will be extracted from created nodes)
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            node_type=node_type,
            node_kwargs=node_kwargs,
            layer_config=layer_config,
            flip_probability=flip_probability,
            grad_stabilization=grad_stabilization,
            grad_target_std=grad_target_std,
            grad_subtract_mean=grad_subtract_mean,
            grad_epsilon=grad_epsilon,
        )

        # Initialize the random mapping
        self._init_mapping(
            ensure_full_input_coverage=ensure_full_input_coverage,
            mapping_indices=mapping_indices,
        )

    def _init_mapping(
        self,
        ensure_full_input_coverage: bool = True,
        mapping_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize random mapping matrix.
        Ensures each input is used at least once per node before any reuse.

        Creates an index tensor of shape (output_size, n) where each entry specifies
        which input index to use. This is more memory efficient than the binary mask.
        """
        # Store current RNG state and set seed for reproducibility
        rng_state = torch.get_rng_state()
        torch.manual_seed(self.seed)

        # Create index mapping: (output_size, n)
        # For each output node and position, store which input index to use
        # Use int16 for memory efficiency (supports up to 32k input features)
        dtype = torch.int16 if self.input_size < 32768 else torch.int32

        if mapping_indices is not None:
            mapping_indices = mapping_indices.to(dtype=dtype)

        elif not ensure_full_input_coverage:
            mapping_indices = torch.multinomial(
                input=torch.ones((self.output_size, self.input_size)),
                num_samples=self.n,
                replacement=self.n > self.input_size,
            ).to(dtype=dtype)

        else:
            node_inputs = self.output_size * self.n
            full_cycles = node_inputs // self.input_size
            remaining_inputs = node_inputs % self.input_size
            all_indices = []

            for _ in range(full_cycles):
                all_indices.append(torch.randperm(self.input_size))

            if remaining_inputs > 0:
                all_indices.append(
                    torch.multinomial(
                        input=torch.ones(self.input_size),
                        num_samples=remaining_inputs,
                        replacement=False,
                    )
                )

            mapping_indices = (
                torch.cat(all_indices).reshape(self.output_size, self.n).to(dtype=dtype)
            )

        # Register as buffer (not a parameter, but saved with model)
        # Shape: (output_size, n) - index mapping (int16/int32)
        # Memory: output_size * n * 2 bytes (vs input_size * output_size * n * 1 byte for mask)
        # For typical case: 1000 * 6 * 2 = 12KB (vs 1568 * 1000 * 6 * 1 = 9.4MB)
        # Reduction: ~99% memory savings for mapping storage!
        self.register_buffer("_mapping_indices", mapping_indices)

        # Restore original RNG state
        torch.set_rng_state(rng_state)

    def get_mapping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get mapped inputs using the fixed random mapping.

        Uses custom CUDA kernel when available for optimal performance.
        Falls back to PyTorch gather operations on CPU or if CUDA extension unavailable.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Mapped inputs of shape (batch_size, output_size, n)
        """
        # Try CUDA kernel first (fastest, eliminates expand + gather overhead)
        if _MAPPING_CUDA_AVAILABLE and x.is_cuda:
            mapped_inputs = mapping_forward_cuda(x, self._mapping_indices, self.input_size)
            if mapped_inputs is not None:
                return mapped_inputs

        # PyTorch fallback - efficient gather-based implementation
        # MEMORY OPTIMIZATION: Use gather with index tensor
        # This is more memory efficient than einsum as it:
        # 1. Doesn't require float conversion of mask
        # 2. Uses optimized indexing kernels
        # 3. Avoids intermediate sparse matmul operations

        batch_size = x.shape[0]

        # Expand indices for batch dimension: (1, output_size, n) -> (batch_size, output_size, n)
        indices_expanded = self._mapping_indices.unsqueeze(0).expand(batch_size, -1, -1)

        # Convert to long for indexing
        indices_long = indices_expanded.long()

        # Gather values: for each (batch, output, pos), get x[batch, indices[output, pos]]
        # x: (batch_size, input_size)
        # We need to gather from input_size dimension using indices
        # Expand x to match output dimension, then gather along input dimension

        # Approach: Use batched index_select equivalent via gather
        # x.unsqueeze(1): (batch_size, 1, input_size)
        # expand to: (batch_size, output_size, input_size)
        # then gather along dim=2 using indices: (batch_size, output_size, n)
        x_expanded = x.unsqueeze(1).expand(-1, self.output_size, -1)
        mapped_inputs = torch.gather(x_expanded, dim=2, index=indices_long)

        return mapped_inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through random mapping and node.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size * output_dim)
        """
        # Validate input dimensions
        self._validate_input_dims(x)

        # Get mapped inputs: (batch_size, output_size, n)
        mapped_inputs = self.get_mapping(x)

        batch_size = mapped_inputs.shape[0]
        output_dim: int = self.nodes[0].output_dim  # pyright: ignore[reportAssignmentType]

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
        """Get the random mapping matrix for inspection (as indices)."""
        # Return the index mapping directly
        return self._mapping_indices.long()  # pyright: ignore[reportCallIssue]

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        flip_str = f", flip_prob={self.flip_probability}" if self.flip_probability > 0 else ""
        grad_str = (
            f", grad_stab={self.grad_stabilization}" if self.grad_stabilization != "none" else ""
        )
        return (
            f"input_size={self.input_size}, output_size={self.output_size}, "
            f"n={self.n}, seed={self.seed}, mapping=random{flip_str}{grad_str}"
        )
