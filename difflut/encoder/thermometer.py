import warnings
from typing import Optional

import torch

from ..registry import register_encoder
from ..utils.warnings import warn_default_value
from .base_encoder import BaseEncoder

# Default number of bits for thermometer encoding
DEFAULT_THERMOMETER_NUM_BITS: int = 1


@register_encoder("thermometer")
class ThermometerEncoder(BaseEncoder):
    """
    Thermometer encoding: represents a value using multiple thresholds.
    Each threshold produces a binary bit, creating a thermometer-like pattern.

    Example: value=0.6 with 3 bits and thresholds [0.25, 0.5, 0.75] -> [1, 1, 0]
    """

    def __init__(self, num_bits: int = DEFAULT_THERMOMETER_NUM_BITS, flatten: bool = True) -> None:
        """
        Args:
            num_bits: Number of threshold bits
            flatten: If True, return 2D (batch_size, input_dim * num_bits);
                     if False, return 3D (batch_size, input_dim, num_bits)
        """
        super().__init__(num_bits=num_bits, flatten=flatten)

        # Note: Warnings for using default values are removed here since parameters
        # are now explicitly provided in configs. Only warn when parameters are
        # truly missing (not provided in kwargs).

        self.thresholds = None

    def _compute_thresholds(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute evenly spaced thresholds between min and max values per feature.

        Args:
            x: Input tensor

        Returns:
            Threshold tensor with shape (num_features, num_bits)
        """
        # Compute min/max per feature
        min_value = x.min(dim=0)[0]
        max_value = x.max(dim=0)[0]

        # Create evenly spaced thresholds
        threshold_indices = torch.arange(1, self.num_bits + 1, device=x.device)
        step_size = (max_value - min_value) / (self.num_bits + 1)

        # Shape: (num_features, num_bits)
        thresholds = min_value.unsqueeze(-1) + threshold_indices.unsqueeze(0) * step_size.unsqueeze(
            -1
        )

        return thresholds

    def fit(self, x: torch.Tensor) -> "ThermometerEncoder":
        """
        Fit the encoder by computing thresholds from the data.

        Args:
            x: Input data tensor

        Returns:
            self for method chaining
        """
        x = self._to_tensor(x)

        # Warn if num_bits is very large (exponential growth in features)
        if self.num_bits > 32:
            warnings.warn(
                f"ThermometerEncoder: Using {self.num_bits} bits will create many features "
                f"({self.num_bits} per input feature). This may lead to memory issues and overfitting. "
                f"Consider using fewer bits (typically 4-8 bits is sufficient).",
                UserWarning,
                stacklevel=2,
            )

        self.thresholds = self._compute_thresholds(x)
        self._is_fitted = True
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using thermometer encoding.

        Args:
            x: Input tensor to encode

        Returns:
            Binary encoded tensor.
            - If flatten=True: shape (batch_size, num_features * num_bits)
            - If flatten=False: shape (batch_size, num_features, num_bits)
        """
        self._check_fitted()
        x = self._to_tensor(x)

        # Add dimension for broadcasting with thresholds
        x_expanded = x.unsqueeze(-1)

        # Compare with thresholds to get binary encoding
        # Shape: (batch_size, num_features, num_bits)
        encoded = (x_expanded > self.thresholds).float()

        # Flatten if requested
        if self.flatten:
            batch_size = encoded.shape[0]
            encoded = encoded.reshape(batch_size, -1)

        return encoded

    # Backward compatibility aliases
    def binarize(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for encode() for backward compatibility."""
        return self.encode(x)

    def get_thresholds(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for _compute_thresholds() for backward compatibility."""
        return self._compute_thresholds(x)

    def __repr__(self) -> str:
        return f"ThermometerEncoder(num_bits={self.num_bits}, flatten={self.flatten})"


@register_encoder("gaussian_thermometer")
class GaussianThermometerEncoder(ThermometerEncoder):
    """
    Gaussian Thermometer encoding: uses Gaussian distribution quantiles for thresholds.
    Thresholds are placed at inverse CDF values assuming normal distribution.

    Better for data that follows a Gaussian distribution.
    """

    def _compute_thresholds(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute thresholds based on Gaussian quantiles per feature.

        Args:
            x: Input tensor

        Returns:
            Threshold tensor with shape (num_features, num_bits)
        """
        # Compute quantiles for Gaussian distribution
        quantile_positions = torch.arange(1, self.num_bits + 1, device=x.device).float() / (
            self.num_bits + 1
        )
        std_skews = torch.distributions.Normal(0, 1).icdf(quantile_positions)

        # Compute mean and std per feature
        mean = x.mean(dim=0)
        std = x.std(dim=0)

        # Compute thresholds: threshold = mean + std_skew * std
        # Shape: (num_features, num_bits)
        thresholds = torch.stack([std_skew * std + mean for std_skew in std_skews], dim=-1)

        return thresholds

    def __repr__(self) -> str:
        return f"GaussianThermometerEncoder(num_bits={self.num_bits}, flatten={self.flatten})"


@register_encoder("distributive_thermometer")
class DistributiveThermometerEncoder(ThermometerEncoder):
    """
    Distributive Thermometer encoding: uses data distribution quantiles for thresholds.
    Thresholds are placed at actual data quantiles, ensuring balanced representation.

    Better for arbitrary data distributions as it adapts to the actual data.
    """

    def _compute_thresholds(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute thresholds based on data distribution quantiles per feature.

        Args:
            x: Input tensor

        Returns:
            Threshold tensor with shape (num_features, num_bits)
        """
        # Sort along the sample dimension (dim=0)
        data_sorted = torch.sort(x, dim=0)[0]
        num_samples = data_sorted.shape[0]

        # Compute indices for quantiles
        indices = torch.tensor(
            [int(num_samples * i / (self.num_bits + 1)) for i in range(1, self.num_bits + 1)],
            device=x.device,
        )

        # Get thresholds at quantile positions
        thresholds = data_sorted[indices]  # Shape: (num_bits, num_features)

        # Permute to (num_features, num_bits)
        thresholds = thresholds.permute(*list(range(1, thresholds.ndim)), 0)

        return thresholds

    def __repr__(self) -> str:
        return f"DistributiveThermometerEncoder(num_bits={self.num_bits}, flatten={self.flatten})"
