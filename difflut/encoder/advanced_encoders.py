import warnings

import numpy as np
import torch

from ..registry import register_encoder
from ..utils.warnings import warn_default_value
from .base_encoder import BaseEncoder

# Default number of bits for advanced encoders (Binary, Gray, etc.)
DEFAULT_ADVANCED_ENCODER_NUM_BITS: int = 8
# Default base for logarithmic encoder
DEFAULT_LOGARITHMIC_BASE: float = 2.0
# Minimum number of bits required for sign-magnitude encoding
MIN_SIGN_MAGNITUDE_BITS: int = 2
# Default number of bits for Gray encoder
DEFAULT_GRAY_ENCODER_NUM_BITS: int = 8
# Default number of bits for OneHot encoder
DEFAULT_ONEHOT_ENCODER_NUM_BITS: int = 8
# Default number of bits for Binary encoder
DEFAULT_BINARY_ENCODER_NUM_BITS: int = 8
# Default number of bits for Sign-Magnitude encoder
DEFAULT_SIGN_MAGNITUDE_NUM_BITS: int = 8


@register_encoder("gray")
class GrayEncoder(BaseEncoder):
    """
    Gray code encoding: adjacent values differ by only one bit.
    Useful for minimizing errors when values change incrementally.

    Example:
        Binary: 0(000), 1(001), 2(010), 3(011), 4(100)
        Gray:   0(000), 1(001), 2(011), 3(010), 4(110)
    """

    def __init__(self, num_bits: int = 8, flatten: bool = True):
        """
        Args:
            num_bits: Number of bits in the encoded representation
            flatten: If True, return 2D (batch_size, input_dim * num_bits);
                     if False, return 3D (batch_size, input_dim, num_bits)
        """
        super().__init__(num_bits=num_bits, flatten=flatten)

        # Warn if using defaults for this specific encoder
        if num_bits == 8:
            warn_default_value("num_bits (GrayEncoder)", num_bits, stacklevel=2)
        if flatten == True:
            warn_default_value("flatten (GrayEncoder)", flatten, stacklevel=2)

        # Don't initialize as None - let register_buffer handle it
        # self.min_value = None
        # self.max_value = None
        self.max_int = (2**num_bits) - 1

    def _binary_to_gray(self, binary: torch.Tensor) -> torch.Tensor:
        """Convert binary representation to Gray code."""
        return binary ^ (binary >> 1)

    def _int_to_gray_bits(self, value: torch.Tensor) -> torch.Tensor:
        """Convert integer values to Gray code bit representation."""
        # Convert to Gray code
        gray = self._binary_to_gray(value.long())

        # Convert to binary bits
        bits = []
        for i in range(self.num_bits - 1, -1, -1):
            bits.append((gray >> i) & 1)

        return torch.stack(bits, dim=-1).float()

    def fit(self, x: torch.Tensor) -> "GrayEncoder":
        """
        Fit the encoder by computing min/max values per feature from the data.

        Args:
            x: Input data tensor

        Returns:
            self for method chaining
        """
        x = self._to_tensor(x)

        min_value = x.min(dim=0)[0]
        max_value = x.max(dim=0)[0]
        # Register or update as buffers so they move with the model via .to(device)
        try:
            # Try to copy if buffers already exist
            self.min_value.copy_(min_value)
            self.max_value.copy_(max_value)
        except AttributeError:
            # First time - register the buffers
            self.register_buffer("min_value", min_value)
            self.register_buffer("max_value", max_value)

        self._is_fitted = True
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using Gray code.

        Args:
            x: Input tensor to encode

        Returns:
            Gray code encoded tensor.
            - If flatten=True: shape (batch_size, num_features * num_bits)
            - If flatten=False: shape (batch_size, num_features, num_bits)
        """
        self._check_fitted()
        x = self._to_tensor(x)

        # Normalize to [0, max_int]
        x_normalized = (x - self.min_value) / (self.max_value - self.min_value + 1e-8)
        x_normalized = torch.clamp(x_normalized, 0, 1)
        x_int = (x_normalized * self.max_int).long()

        # Convert to Gray code bits
        encoded = self._int_to_gray_bits(x_int)

        # Flatten if requested
        if self.flatten:
            batch_size = encoded.shape[0]
            encoded = encoded.reshape(batch_size, -1)

        return encoded

    def __repr__(self) -> str:
        return f"GrayEncoder(num_bits={self.num_bits}, flatten={self.flatten})"


@register_encoder("onehot")
class OneHotEncoder(BaseEncoder):
    """
    One-hot encoding: divides the range into bins and uses one-hot representation.
    Each bin is represented by a single active bit.

    Useful for categorical-like binning of continuous values.
    """

    def __init__(self, num_bits: int = 8, flatten: bool = True):
        """
        Args:
            num_bits: Number of bins (one per bit)
            flatten: If True, return 2D (batch_size, input_dim * num_bits);
                     if False, return 3D (batch_size, input_dim, num_bits)
        """
        super().__init__(num_bits=num_bits, flatten=flatten)

        # Warn if using defaults for this specific encoder
        if num_bits == 8:
            warn_default_value("num_bits (OneHotEncoder)", num_bits, stacklevel=2)
        if flatten == True:
            warn_default_value("flatten (OneHotEncoder)", flatten, stacklevel=2)

        # Don't initialize as None - let register_buffer handle it
        # self.bin_edges = None

    def fit(self, x: torch.Tensor) -> "OneHotEncoder":
        """
        Fit the encoder by computing bin edges per feature from the data.

        Args:
            x: Input data tensor

        Returns:
            self for method chaining
        """
        x = self._to_tensor(x)

        min_val = x.min(dim=0)[0]
        max_val = x.max(dim=0)[0]
        # Create bin edges for each feature
        # Shape: (num_features, num_bits + 1)
        bin_edges = torch.stack(
            [
                torch.linspace(min_val[i], max_val[i], self.num_bits + 1, device=x.device)
                for i in range(x.shape[1])
            ],
            dim=0,
        )
        # Register or update bin_edges as a buffer so it moves with the model via .to(device)
        try:
            # Try to copy if buffer already exists
            self.bin_edges.copy_(bin_edges)
        except AttributeError:
            # First time - register the buffer
            self.register_buffer("bin_edges", bin_edges)

        self._is_fitted = True
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using one-hot encoding.

        Args:
            x: Input tensor to encode

        Returns:
            One-hot encoded tensor.
            - If flatten=True: shape (batch_size, num_features * num_bits)
            - If flatten=False: shape (batch_size, num_features, num_bits)
        """
        self._check_fitted()
        x = self._to_tensor(x)

        # Digitize each feature
        batch_size = x.shape[0]
        encoded = torch.zeros(batch_size, x.shape[1], self.num_bits, device=x.device)

        for i in range(x.shape[1]):
            bins = torch.bucketize(x[:, i], self.bin_edges[i][1:-1])
            bins = torch.clamp(bins, 0, self.num_bits - 1)
            encoded[:, i, :] = torch.nn.functional.one_hot(bins, num_classes=self.num_bits).float()

        # Flatten if requested
        if self.flatten:
            encoded = encoded.reshape(batch_size, -1)

        return encoded

    def __repr__(self) -> str:
        return f"OneHotEncoder(num_bits={self.num_bits}, flatten={self.flatten})"


@register_encoder("binary")
class BinaryEncoder(BaseEncoder):
    """
    Standard binary encoding: converts normalized values to binary representation.

    Example with 3 bits: 0->000, 1->001, 2->010, 3->011, 4->100, etc.
    """

    def __init__(self, num_bits: int = 8, flatten: bool = True):
        """
        Args:
            num_bits: Number of bits in the encoded representation
            flatten: If True, return 2D (batch_size, input_dim * num_bits);
                     if False, return 3D (batch_size, input_dim, num_bits)
        """
        super().__init__(num_bits=num_bits, flatten=flatten)

        # Warn if using defaults for this specific encoder
        if num_bits == 8:
            warn_default_value("num_bits (BinaryEncoder)", num_bits, stacklevel=2)
        if flatten == True:
            warn_default_value("flatten (BinaryEncoder)", flatten, stacklevel=2)

        # Don't initialize as None - let register_buffer handle it
        # self.min_value = None
        # self.max_value = None
        self.max_int = (2**num_bits) - 1

    def fit(self, x: torch.Tensor) -> "BinaryEncoder":
        """
        Fit the encoder by computing min/max values per feature from the data.

        Args:
            x: Input data tensor

        Returns:
            self for method chaining
        """
        x = self._to_tensor(x)

        min_value = x.min(dim=0)[0]
        max_value = x.max(dim=0)[0]
        # Register or update as buffers so they move with the model via .to(device)
        try:
            # Try to copy if buffers already exist
            self.min_value.copy_(min_value)
            self.max_value.copy_(max_value)
        except AttributeError:
            # First time - register the buffers
            self.register_buffer("min_value", min_value)
            self.register_buffer("max_value", max_value)

        self._is_fitted = True
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using standard binary encoding.

        Args:
            x: Input tensor to encode

        Returns:
            Binary encoded tensor.
            - If flatten=True: shape (batch_size, num_features * num_bits)
            - If flatten=False: shape (batch_size, num_features, num_bits)
        """
        self._check_fitted()
        x = self._to_tensor(x)

        # Normalize to [0, max_int]
        x_normalized = (x - self.min_value) / (self.max_value - self.min_value + 1e-8)
        x_normalized = torch.clamp(x_normalized, 0, 1)
        x_int = (x_normalized * self.max_int).long()

        # Convert to binary bits
        bits = []
        for i in range(self.num_bits - 1, -1, -1):
            bits.append((x_int >> i) & 1)

        encoded = torch.stack(bits, dim=-1).float()

        # Flatten if requested
        if self.flatten:
            batch_size = encoded.shape[0]
            encoded = encoded.reshape(batch_size, -1)

        return encoded

    def __repr__(self) -> str:
        return f"BinaryEncoder(num_bits={self.num_bits}, flatten={self.flatten})"


@register_encoder("sign_magnitude")
class SignMagnitudeEncoder(BaseEncoder):
    """
    Sign-magnitude encoding: one bit for sign, remaining bits for magnitude.
    Useful when preserving the sign of values is important.

    Example with 4 bits: +5 -> 0101, -5 -> 1101
    """

    def __init__(self, num_bits: int = 8, flatten: bool = True):
        """
        Args:
            num_bits: Number of bits (1 for sign, rest for magnitude)
            flatten: If True, return 2D (batch_size, input_dim * num_bits);
                     if False, return 3D (batch_size, input_dim, num_bits)
        """
        super().__init__(num_bits=num_bits, flatten=flatten)

        # Warn if using defaults for this specific encoder
        if num_bits == 8:
            warn_default_value("num_bits (SignMagnitudeEncoder)", num_bits, stacklevel=2)
        if flatten == True:
            warn_default_value("flatten (SignMagnitudeEncoder)", flatten, stacklevel=2)

        assert num_bits >= 2, "num_bits must be at least 2 for sign-magnitude encoding"
        # Don't initialize as None - let register_buffer handle it
        # self.max_abs_value = None
        self.magnitude_bits = num_bits - 1
        self.max_int = (2**self.magnitude_bits) - 1

    def fit(self, x: torch.Tensor) -> "SignMagnitudeEncoder":
        """
        Fit the encoder by computing max absolute value per feature from the data.

        Args:
            x: Input data tensor

        Returns:
            self for method chaining
        """
        x = self._to_tensor(x)

        max_abs_value = x.abs().max(dim=0)[0]
        # Register or update as buffer so it moves with the model via .to(device)
        try:
            # Try to copy if buffer already exists
            self.max_abs_value.copy_(max_abs_value)
        except AttributeError:
            # First time - register the buffer
            self.register_buffer("max_abs_value", max_abs_value)

        self._is_fitted = True
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using sign-magnitude encoding.

        Args:
            x: Input tensor to encode

        Returns:
            Sign-magnitude encoded tensor.
            - If flatten=True: shape (batch_size, num_features * num_bits)
            - If flatten=False: shape (batch_size, num_features, num_bits)
        """
        self._check_fitted()
        x = self._to_tensor(x)

        # Extract sign bit
        sign_bit = (x < 0).float()

        # Normalize magnitude to [0, max_int]
        magnitude = torch.abs(x)
        magnitude_normalized = magnitude / (self.max_abs_value + 1e-8)
        magnitude_normalized = torch.clamp(magnitude_normalized, 0, 1)
        magnitude_int = (magnitude_normalized * self.max_int).long()

        # Convert magnitude to binary bits
        bits = [sign_bit]
        for i in range(self.magnitude_bits - 1, -1, -1):
            bits.append(((magnitude_int >> i) & 1).float())

        encoded = torch.stack(bits, dim=-1)

        # Flatten if requested
        if self.flatten:
            batch_size = encoded.shape[0]
            encoded = encoded.reshape(batch_size, -1)

        return encoded

    def __repr__(self) -> str:
        return f"SignMagnitudeEncoder(num_bits={self.num_bits}, flatten={self.flatten})"


@register_encoder("logarithmic")
class LogarithmicEncoder(BaseEncoder):
    """
    Logarithmic encoding: applies log transformation before binary encoding.
    Useful for data with wide dynamic ranges or exponential distributions.

    Provides better resolution for small values and compresses large values.
    """

    def __init__(self, num_bits: int = 8, base: float = 2.0, flatten: bool = True):
        """
        Args:
            num_bits: Number of bits in the encoded representation
            base: Base of the logarithm (default: 2.0)
            flatten: If True, return 2D (batch_size, input_dim * num_bits);
                     if False, return 3D (batch_size, input_dim, num_bits)
        """
        super().__init__(num_bits=num_bits, flatten=flatten)

        # Warn if using defaults for this specific encoder
        if num_bits == 8:
            warn_default_value("num_bits (LogarithmicEncoder)", num_bits, stacklevel=2)
        if base == 2.0:
            warn_default_value("base (LogarithmicEncoder)", base, stacklevel=2)
        if flatten == True:
            warn_default_value("flatten (LogarithmicEncoder)", flatten, stacklevel=2)

        assert base > 0 and base != 1, "base must be positive and not equal to 1"
        self.base = base
        # Don't initialize as None - let register_buffer handle it
        # self.min_log = None
        # self.max_log = None
        # self.offset = None  # To handle negative values
        self.max_int = (2**num_bits) - 1

    def fit(self, x: torch.Tensor) -> "LogarithmicEncoder":
        """
        Fit the encoder by computing log range per feature from the data.

        Args:
            x: Input data tensor

        Returns:
            self for method chaining
        """
        x = self._to_tensor(x)

        # Add offset to make all values positive (per-feature)
        min_val = x.min(dim=0)[0]
        offset = torch.where(min_val < 0, -min_val + 1, torch.zeros_like(min_val))

        x_shifted = x + offset

        # Compute log range
        x_log = torch.log(x_shifted + 1e-8) / torch.log(torch.tensor(self.base))

        min_log = x_log.min(dim=0)[0]
        max_log = x_log.max(dim=0)[0]

        # Register or update as buffers so they move with the model via .to(device)
        try:
            # Try to copy if buffers already exist
            self.offset.copy_(offset)
            self.min_log.copy_(min_log)
            self.max_log.copy_(max_log)
        except AttributeError:
            # First time - register the buffers
            self.register_buffer("offset", offset)
            self.register_buffer("min_log", min_log)
            self.register_buffer("max_log", max_log)

        self._is_fitted = True
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using logarithmic encoding.

        Args:
            x: Input tensor to encode

        Returns:
            Logarithmically encoded tensor.
            - If flatten=True: shape (batch_size, num_features * num_bits)
            - If flatten=False: shape (batch_size, num_features, num_bits)
        """
        self._check_fitted()
        x = self._to_tensor(x)

        # Apply offset and log transform
        x_shifted = x + self.offset
        x_log = torch.log(x_shifted + 1e-8) / torch.log(torch.tensor(self.base))

        # Normalize to [0, max_int]
        x_normalized = (x_log - self.min_log) / (self.max_log - self.min_log + 1e-8)
        x_normalized = torch.clamp(x_normalized, 0, 1)
        x_int = (x_normalized * self.max_int).long()

        # Convert to binary bits
        bits = []
        for i in range(self.num_bits - 1, -1, -1):
            bits.append(((x_int >> i) & 1).float())

        encoded = torch.stack(bits, dim=-1)

        # Flatten if requested
        if self.flatten:
            batch_size = encoded.shape[0]
            encoded = encoded.reshape(batch_size, -1)

        return encoded

    def __repr__(self) -> str:
        return f"LogarithmicEncoder(num_bits={self.num_bits}, base={self.base}, flatten={self.flatten})"
