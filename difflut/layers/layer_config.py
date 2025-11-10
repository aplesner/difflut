"""
Configuration class for layer training parameters.

This module provides a typed, maintainable way to specify layer training
parameters like bit flip augmentation and gradient stabilization, following
the same pattern as ConvolutionConfig and NodeConfig.
"""

from dataclasses import dataclass

import torch


@dataclass
class LayerConfig:
    """
    Configuration for layer training parameters (bit flip and gradient stabilization).

    This class provides type-safe configuration for training augmentation and
    gradient control features that are shared across all LUT layers via LUTLayerMixin.

    Parameters:
        flip_probability: Probability of flipping bits during training [0, 1]
            - 0.0: No bit flipping (default)
            - 0.1: Flip 10% of bits for robustness training

        grad_stabilization: Gradient stabilization mode
            - 'none': No stabilization (default)
            - 'layerwise': Normalize across all layer gradients
            - 'batchwise': Normalize per batch sample

        grad_target_std: Target standard deviation for gradient rescaling
            - Default: 1.0 (unit variance)
            - Used when grad_stabilization is not 'none'

        grad_subtract_mean: Whether to center gradients (subtract mean)
            - Default: False
            - If True, gradients are mean-centered before rescaling

        grad_epsilon: Small constant for numerical stability
            - Default: 1e-8
            - Used in variance calculation to avoid division by zero

    Example:
        ```python
        # Create configuration for robust training
        layer_config = LayerConfig(
            flip_probability=0.1,
            grad_stabilization='layerwise',
            grad_target_std=1.0
        )

        # Use in layer
        layer = RandomLayer(
            input_size=100,
            output_size=50,
            node_type=DWNNode,
            node_kwargs=node_config,
            layer_config=layer_config  # Pass config object
        )

        # Or use individual parameters (backward compatible)
        layer = RandomLayer(
            input_size=100,
            output_size=50,
            node_type=DWNNode,
            node_kwargs=node_config,
            flip_probability=0.1,
            grad_stabilization='layerwise'
        )
        ```
    """

    flip_probability: float = 0.0
    grad_stabilization: str = "none"
    grad_target_std: float = 1.0
    grad_subtract_mean: bool = False
    grad_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate flip_probability
        if not isinstance(self.flip_probability, (int, float)) or not (
            0.0 <= self.flip_probability <= 1.0
        ):
            raise ValueError(
                f"flip_probability must be a float in [0, 1], got {self.flip_probability}. "
                f"Example: flip_probability=0.1 for 10% bit flipping during training."
            )

        # Validate grad_stabilization
        valid_grad_modes = ["none", "layerwise", "batchwise"]
        if self.grad_stabilization not in valid_grad_modes:
            raise ValueError(
                f"grad_stabilization must be one of {valid_grad_modes}, got '{self.grad_stabilization}'. "
                f"'layerwise': normalize per layer, 'batchwise': normalize per batch sample, 'none': disabled"
            )

        # Validate grad_target_std
        if not isinstance(self.grad_target_std, (int, float)) or self.grad_target_std <= 0:
            raise ValueError(
                f"grad_target_std must be a positive number, got {self.grad_target_std}. "
                f"Example: grad_target_std=1.0 for unit variance"
            )

        # Validate grad_epsilon
        if not isinstance(self.grad_epsilon, (int, float)) or self.grad_epsilon <= 0:
            raise ValueError(
                f"grad_epsilon must be a positive number, got {self.grad_epsilon}. "
                f"Used for numerical stability in variance calculation"
            )

    def copy(self) -> "LayerConfig":
        """Create a copy of this configuration."""
        return LayerConfig(
            flip_probability=self.flip_probability,
            grad_stabilization=self.grad_stabilization,
            grad_target_std=self.grad_target_std,
            grad_subtract_mean=self.grad_subtract_mean,
            grad_epsilon=self.grad_epsilon,
        )

    def __repr__(self) -> str:
        """String representation showing all parameters."""
        params = []
        if self.flip_probability > 0:
            params.append(f"flip_probability={self.flip_probability}")
        if self.grad_stabilization != "none":
            params.append(f"grad_stabilization='{self.grad_stabilization}'")
            params.append(f"grad_target_std={self.grad_target_std}")
            if self.grad_subtract_mean:
                params.append(f"grad_subtract_mean={self.grad_subtract_mean}")

        if not params:
            return "LayerConfig(default)"

        return f"LayerConfig({', '.join(params)})"


@dataclass
class GroupedInputConfig:
    """
    Configuration for grouped layers.

    Here the input features are divided into k groups of equal size, and each node only receives inputs from a single group.

    Parameters:
        n_groups: Number of input groups
            - Default: 1 (no grouping)
            - Example: n_groups=4 divides inputs into 4 groups for group-wise processing
    """

    n_groups: int
    mapping_indices: torch.Tensor
    luts_per_tree: int

    def __init__(
        self,
        n_groups: int,
        input_size: int,
        output_trees: int,
        luts_per_tree: int,
        bits_per_node: int,
        seed: int,
        ensure_full_coverage: bool = True,
    ) -> None:
        if n_groups < 1:
            raise ValueError(
                f"n_groups must be >= 1, got {n_groups}. Example: n_groups=4 for 4 input groups."
            )
        self.n_groups = n_groups
        self.luts_per_tree = luts_per_tree

        # The group size is the size of the receptive field. And the number of groups is the number of input channels/features.
        # We need to create mapping indices for output_size = output_trees * luts_per_tree
        output_size = output_trees * luts_per_tree

        # Store current RNG state and set seed for reproducibility
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)

        group_size = input_size // n_groups
        # Create mapping indices of shape (output_size, n) with indices up to group_size
        mapping_indices = torch.multinomial(
            input=torch.ones((output_size, group_size)),
            num_samples=bits_per_node,
            replacement=bits_per_node > group_size,
        )

        if not ensure_full_coverage:
            # Offset indices for each group
            group_offsets = torch.randint(low=0, high=n_groups, size=(output_size, 1))
            group_offsets *= group_size
        else:
            full_cycles = output_size // n_groups
            remainder = output_size % n_groups
            group_offsets = []
            for _ in range(full_cycles):
                group_offsets.append(torch.randperm(n_groups))
            if remainder > 0:
                group_offsets.append(
                    torch.multinomial(
                        input=torch.ones((n_groups,)),
                        num_samples=remainder,
                        replacement=False,
                    )
                )
            group_offsets = torch.cat(group_offsets).unsqueeze(1) * group_size

        mapping_indices += group_offsets

        self.mapping_indices = mapping_indices

        torch.set_rng_state(rng_state)

    def get_mapping_indices(self, chunk_start: int, chunk_end: int) -> torch.Tensor:
        """Get mapping indices for a specific chunk of output nodes. Chunk indices are relative to the output trees."""
        start_idx = chunk_start * self.luts_per_tree
        end_idx = chunk_end * self.luts_per_tree
        return self.mapping_indices[start_idx:end_idx, :]
