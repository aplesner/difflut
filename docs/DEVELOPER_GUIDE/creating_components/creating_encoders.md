# Creating Custom Encoders

Learn how to implement and register custom DiffLUT encoders.

---

## Overview

Encoders transform continuous input values into discrete binary representations suitable for LUT indexing. All custom encoders:

1. **Extend** `BaseEncoder`
2. **Register** using `@register_encoder` decorator
3. **Implement** `fit()` and `forward()` methods
4. **Optionally add** CUDA support for performance

### Key Concepts

- **Fitting**: Learn quantization ranges from training data
- **Encoding**: Convert continuous values to binary representations
- **Feature-wise vs. Global**: Per-feature or global quantization
- **Flattening**: Reshape encoded output for layer input

---

## Type-Safe Configuration

DiffLUT uses type-safe configuration for encoders:

```python
from difflut.encoder import BaseEncoder
from difflut import register_encoder

# Module defaults (at top level)
DEFAULT_ENCODER_NUM_BITS: int = 8
DEFAULT_ENCODER_FEATURE_WISE: bool = True

@register_encoder('my_custom_encoder')
class MyCustomEncoder(BaseEncoder):
    """Custom encoder with clear documentation."""
    
    def __init__(
        self,
        num_bits: int = DEFAULT_ENCODER_NUM_BITS,
        feature_wise: bool = DEFAULT_ENCODER_FEATURE_WISE,
        flatten: bool = True
    ) -> None:
        """Initialize encoder with type-safe parameters."""
        super().__init__()
        self.num_bits = num_bits
        self.feature_wise = feature_wise
        self.flatten = flatten
        self._is_fitted = False
```

**Why this pattern?**
- Type safety with IDE autocomplete
- Module-level constants for defaults with clear traceability
- `_is_fitted` flag prevents use before fitting
- Clear separation of initialization and fitting

---

## Complete Encoder Implementation

### Step 1: Define Module Defaults

```python
import torch
import torch.nn as nn
from typing import Optional
from difflut.encoder import BaseEncoder
from difflut.utils.warnings import warn_default_value
from difflut import register_encoder

# Module-level defaults (CAPITALS convention)
DEFAULT_ENCODER_NUM_BITS: int = 8
DEFAULT_ENCODER_FEATURE_WISE: bool = True
DEFAULT_ENCODER_FLATTEN: bool = True
```

### Step 2: Implement BaseEncoder

```python
@register_encoder('thermometer_custom')
class ThermometerCustomEncoder(BaseEncoder):
    """
    Thermometer code encoder with adaptive thresholds.
    
    Transforms continuous values to thermometer (unary) binary code.
    Example: value 5.7 → threshold crossing at indices → [1,1,1,1,1,0,0,0]
    
    Each bit represents: "Is value >= threshold_i?"
    Thermometer codes are robust to bit errors.
    """
    
    def __init__(
        self,
        num_bits: int = DEFAULT_ENCODER_NUM_BITS,
        feature_wise: bool = DEFAULT_ENCODER_FEATURE_WISE,
        flatten: bool = DEFAULT_ENCODER_FLATTEN,
        quantile_spacing: str = 'linear',  # 'linear' or 'log'
    ) -> None:
        """
        Initialize thermometer encoder.
        
        Args:
            num_bits: Number of quantization levels (thresholds)
            feature_wise: If True, learn separate thresholds per feature
            flatten: If True, flatten output to (batch, features*num_bits)
            quantile_spacing: How to space quantile thresholds
        """
        super().__init__()
        
        # Validate parameters
        if num_bits < 1:
            raise ValueError(f"num_bits must be >= 1, got {num_bits}")
        
        self.num_bits = num_bits
        self.feature_wise = feature_wise
        self.flatten = flatten
        self.quantile_spacing = quantile_spacing
        
        # Will be set during fit()
        self.register_buffer('thresholds', None)
        self._is_fitted = False
    
    def fit(self, data: torch.Tensor) -> 'ThermometerCustomEncoder':
        """
        Learn quantile-based thresholds from training data.
        
        Args:
            data: Training data of shape (batch, features)
        
        Returns:
            self (for chaining)
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2D data, got {data.ndim}D")
        
        batch_size, num_features = data.shape
        
        if self.feature_wise:
            # Learn separate thresholds per feature
            thresholds = []
            for feat_idx in range(num_features):
                feat_data = data[:, feat_idx]
                
                # Compute quantiles
                quantiles = torch.linspace(0, 1, self.num_bits + 2)[1:-1]
                feat_thresholds = torch.quantile(feat_data, quantiles)
                thresholds.append(feat_thresholds)
            
            # Stack: (num_features, num_bits)
            self.thresholds = torch.stack(thresholds)
        else:
            # Global thresholds across all data
            flat_data = data.flatten()
            quantiles = torch.linspace(0, 1, self.num_bits + 2)[1:-1]
            self.thresholds = torch.quantile(flat_data, quantiles)
            # Add feature dimension: (1, num_bits)
            self.thresholds = self.thresholds.unsqueeze(0)
        
        self._is_fitted = True
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode data to thermometer code.
        
        Args:
            x: Input data of shape (batch, features)
        
        Returns:
            Encoded data of shape (batch, features, num_bits) or
            (batch, features*num_bits) if flatten=True
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Encoder not fitted. Call .fit(data) on training data first."
            )
        
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input, got {x.ndim}D")
        
        batch_size, num_features = x.shape
        device = x.device
        
        # Move thresholds to same device as input
        thresholds = self.thresholds.to(device)
        
        if self.feature_wise:
            # (batch, features, 1) >= (features, num_bits)
            # Broadcasting: compare each feature against its thresholds
            encoded = (x.unsqueeze(2) >= thresholds.unsqueeze(0)).float()
            # Shape: (batch, features, num_bits)
        else:
            # Global thresholds: compare all values against same thresholds
            encoded = (x.unsqueeze(2) >= thresholds).float()
            # Shape: (batch, features, num_bits)
        
        if self.flatten:
            # Reshape to (batch, features*num_bits)
            encoded = encoded.reshape(batch_size, -1)
        
        return encoded
```

### Step 3: Register and Use

```python
from difflut.registry import REGISTRY

# Your encoder is automatically registered by the decorator
encoder_class = REGISTRY.get_encoder('thermometer_custom')
print(f"✓ Encoder registered: {encoder_class}")

# Create instance
encoder = ThermometerCustomEncoder(num_bits=8, feature_wise=True)

# Fit on training data
train_data = torch.randn(1000, 784)
encoder.fit(train_data)

# Encode new data
test_data = torch.randn(32, 784)
encoded = encoder(test_data)
print(f"Encoded shape: {encoded.shape}")  # (32, 784*8) if flatten=True
```

---

## Advanced Examples

### Example 1: Adaptive Resolution Encoder

```python
@register_encoder('adaptive_resolution')
class AdaptiveResolutionEncoder(BaseEncoder):
    """Encoder that adapts bit resolution per feature based on variance."""
    
    def __init__(self, base_bits: int = 4, max_bits: int = 10):
        super().__init__()
        self.base_bits = base_bits
        self.max_bits = max_bits
        self.register_buffer('bits_per_feature', None)
        self._is_fitted = False
    
    def fit(self, data: torch.Tensor) -> 'AdaptiveResolutionEncoder':
        """Assign more bits to high-variance features."""
        batch_size, num_features = data.shape
        
        # Compute variance per feature
        variances = data.var(dim=0)
        
        # Normalize to [base_bits, max_bits]
        var_min, var_max = variances.min(), variances.max()
        if var_max > var_min:
            normalized = (variances - var_min) / (var_max - var_min)
        else:
            normalized = torch.ones_like(variances)
        
        bits = (self.base_bits + normalized * (self.max_bits - self.base_bits)).round()
        self.bits_per_feature = bits.long()
        
        self._is_fitted = True
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with variable resolution per feature."""
        if not self._is_fitted:
            raise RuntimeError("Encoder not fitted")
        
        batch_size, num_features = x.shape
        encoded_parts = []
        
        for feat_idx in range(num_features):
            num_bits = self.bits_per_feature[feat_idx].item()
            feat_data = x[:, feat_idx:feat_idx+1]  # (batch, 1)
            
            # Simple quantization
            min_val = feat_data.min()
            max_val = feat_data.max()
            normalized = (feat_data - min_val) / (max_val - min_val + 1e-8)
            quantized = (normalized * (2**num_bits - 1)).round()
            
            # Convert to binary
            binary = torch.stack([
                ((quantized.squeeze(1).long() >> i) & 1).float()
                for i in range(num_bits)
            ], dim=1)
            
            encoded_parts.append(binary)
        
        return torch.cat(encoded_parts, dim=1)
```

### Example 2: Gray Code Encoder

```python
@register_encoder('gray_code')
class GrayCodeEncoder(BaseEncoder):
    """Gray code (reflected binary) encoder for robust bit transitions."""
    
    def __init__(self, num_bits: int = 8):
        super().__init__()
        self.num_bits = num_bits
        self._is_fitted = False
    
    def fit(self, data: torch.Tensor) -> 'GrayCodeEncoder':
        """Gray code doesn't require fitting."""
        self._is_fitted = True
        return self
    
    @staticmethod
    def _binary_to_gray(binary: torch.Tensor) -> torch.Tensor:
        """Convert binary to Gray code."""
        gray = binary.clone()
        for i in range(1, binary.shape[-1]):
            gray[..., i] = gray[..., i-1] ^ binary[..., i]
        return gray
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to Gray code."""
        if not self._is_fitted:
            raise RuntimeError("Encoder not fitted")
        
        batch_size, num_features = x.shape
        
        # Normalize to [0, 2^num_bits - 1]
        x_min = x.min()
        x_max = x.max()
        normalized = (x - x_min) / (x_max - x_min + 1e-8)
        quantized = (normalized * (2**self.num_bits - 1)).round().long()
        
        # Convert to binary
        binary = torch.stack([
            ((quantized >> i) & 1).float()
            for i in range(self.num_bits)
        ], dim=2)  # (batch, features, num_bits)
        
        # Convert to Gray code
        gray = self._binary_to_gray(binary)
        
        # Flatten
        return gray.reshape(batch_size, -1)
```

---

## Testing Your Encoder

Create `tests/test_encoders/test_my_encoder.py`:

```python
import torch
import pytest
from difflut.encoder import MyCustomEncoder
from difflut.registry import REGISTRY

class TestMyCustomEncoder:
    
    def test_registration(self):
        """Verify encoder is registered."""
        encoder = REGISTRY.get_encoder('my_custom_encoder')
        assert encoder is not None
    
    def test_fit_initializes_params(self):
        """Test fit() initializes encoder parameters."""
        encoder = MyCustomEncoder(num_bits=8)
        data = torch.randn(100, 10)
        
        encoder.fit(data)
        assert encoder._is_fitted
        assert encoder.thresholds is not None
    
    def test_forward_requires_fit(self):
        """Test forward() raises before fitting."""
        encoder = MyCustomEncoder(num_bits=8)
        data = torch.randn(32, 10)
        
        with pytest.raises(RuntimeError):
            encoder(data)
    
    def test_output_shape_flatten(self):
        """Test output shape with flatten=True."""
        encoder = MyCustomEncoder(num_bits=8, flatten=True)
        train_data = torch.randn(100, 10)
        test_data = torch.randn(32, 10)
        
        encoder.fit(train_data)
        encoded = encoder(test_data)
        
        assert encoded.shape == (32, 10*8)
    
    def test_output_shape_no_flatten(self):
        """Test output shape with flatten=False."""
        encoder = MyCustomEncoder(num_bits=8, flatten=False)
        train_data = torch.randn(100, 10)
        test_data = torch.randn(32, 10)
        
        encoder.fit(train_data)
        encoded = encoder(test_data)
        
        assert encoded.shape == (32, 10, 8)
    
    def test_device_transfer(self, device):
        """Test encoder works after device transfer."""
        encoder = MyCustomEncoder(num_bits=8).to(device)
        train_data = torch.randn(100, 10, device=device)
        test_data = torch.randn(32, 10, device=device)
        
        encoder.fit(train_data)
        encoded = encoder(test_data)
        
        assert encoded.device == device
    
    def test_binary_output(self):
        """Test encoded output is binary [0, 1]."""
        encoder = MyCustomEncoder(num_bits=8)
        train_data = torch.randn(100, 10)
        test_data = torch.randn(32, 10)
        
        encoder.fit(train_data)
        encoded = encoder(test_data)
        
        assert torch.all((encoded == 0) | (encoded == 1))
```

---

## Key Patterns

1. **Module-Level Defaults**: Use CAPITALS for module constants
2. **Type Hints**: Full PEP 484 type hints for IDE support
3. **Fitting**: Separate `fit()` from `__init__()`
4. **Validation**: Check inputs and state in `forward()`
5. **Device Safety**: Move buffers to input device in `forward()`
6. **Docstrings**: NumPy format with full parameter documentation
7. **Registration**: Use `@register_encoder` decorator

---

## Next Steps

1. **Review existing encoders** in `difflut/encoder/` for patterns
2. **Implement fit()** to learn from training data
3. **Implement forward()** to encode new data
4. **Add comprehensive tests** following test patterns
5. **Consider edge cases**: empty data, single feature, device transfers
6. **Integrate with pipelines** via REGISTRY

---

## Resources

- **Base Class**: `difflut/encoder/base_encoder.py`
- **Examples**: `difflut/encoder/thermometer.py`, `difflut/encoder/advanced_encoders.py`
- **Tests**: `tests/test_encoders/`
- **User Guide**: [Encoders Guide](../../USER_GUIDE/components/encoders.md)
- **Registry**: [Registry & Pipeline Guide](../../USER_GUIDE/registry_pipeline.md)
