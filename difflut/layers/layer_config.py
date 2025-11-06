"""
Configuration class for layer training parameters.

This module provides a typed, maintainable way to specify layer training
parameters like bit flip augmentation and gradient stabilization, following
the same pattern as ConvolutionConfig and NodeConfig.
"""

from dataclasses import dataclass
from typing import Optional


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
    grad_stabilization: str = 'none'
    grad_target_std: float = 1.0
    grad_subtract_mean: bool = False
    grad_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate flip_probability
        if not isinstance(self.flip_probability, (int, float)) or not (0.0 <= self.flip_probability <= 1.0):
            raise ValueError(
                f"flip_probability must be a float in [0, 1], got {self.flip_probability}. "
                f"Example: flip_probability=0.1 for 10% bit flipping during training."
            )

        # Validate grad_stabilization
        valid_grad_modes = ['none', 'layerwise', 'batchwise']
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

    def copy(self) -> 'LayerConfig':
        """Create a copy of this configuration."""
        return LayerConfig(
            flip_probability=self.flip_probability,
            grad_stabilization=self.grad_stabilization,
            grad_target_std=self.grad_target_std,
            grad_subtract_mean=self.grad_subtract_mean,
            grad_epsilon=self.grad_epsilon
        )

    def __repr__(self) -> str:
        """String representation showing all parameters."""
        params = []
        if self.flip_probability > 0:
            params.append(f"flip_probability={self.flip_probability}")
        if self.grad_stabilization != 'none':
            params.append(f"grad_stabilization='{self.grad_stabilization}'")
            params.append(f"grad_target_std={self.grad_target_std}")
            if self.grad_subtract_mean:
                params.append(f"grad_subtract_mean={self.grad_subtract_mean}")

        if not params:
            return "LayerConfig(default)"

        return f"LayerConfig({', '.join(params)})"
