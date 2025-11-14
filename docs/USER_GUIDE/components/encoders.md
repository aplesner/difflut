# Encoders Guide

Encoders transform continuous input values into discrete binary representations suitable for LUT indexing. All encoders inherit from `BaseEncoder` and provide a consistent API.

---

## Table of Contents
1. [Basic Usage](#basic-usage)
2. [Available Encoders](#available-encoders)
3. [Common Parameters](#common-parameters)
4. [Encoder Examples](#encoder-examples)
5. [Fitting Best Practices](#fitting-best-practices)

---

## Basic Usage

All encoders follow the same three-step pattern:

```python
from difflut.encoder import ThermometerEncoder
import torch

# 1. Create encoder
encoder = ThermometerEncoder(num_bits=8, flatten=True)

# 2. Fit to data (learns min/max ranges)
train_data = torch.randn(1000, 784)
encoder.fit(train_data)

# 3. Encode inputs
x = torch.randn(32, 784)      # (batch_size, features)
encoded = encoder(x)           # (batch_size, features * num_bits) if flatten=True

# 4. Use in model
from difflut.layers import RandomLayer
layer = RandomLayer(input_size=encoded.shape[1], ...)
output = layer(encoded)
```

---

## Available Encoders

| Encoder | Description | Best For | Output |
|---------|-------------|----------|--------|
| `ThermometerEncoder` | Unary encoding (threshold-based) | Smooth, interpretable discretization | Sparse unary |
| `DistributiveThermometerEncoder` | Distributive thermometer | Handling value distributions | Distributed unary |
| `GrayEncoder` | Gray code (minimal Hamming distance) | Minimizing bit flips between neighbors | Gray code |
| `BinaryEncoder` | Standard binary encoding | Compact representation | Binary |
| `GaussianThermometerEncoder` | Gaussian basis functions | Smooth transitions with continuous-like encoding | Gaussian thermometer |
| `OneHotEncoder` | One-hot encoding | Sparse, interpretable bins | One-hot |
| `SignMagnitudeEncoder` | Sign-magnitude representation | Signed values with clear positive/negative | Sign + magnitude |
| `LogarithmicEncoder` | Logarithmic scaling | Large value ranges, exponential data | Logarithmic |

---

## Common Parameters

All encoders support the following parameters:

### Core Parameters

- **`num_bits`** (int): Number of bits per feature (resolution)
  - Higher values: Better precision, larger output
  - Lower values: Faster, more compact
  - Typical range: 3-8 bits

- **`flatten`** (bool, default=True): Output format
  - `True`: 2D output `(batch_size, features * num_bits)` - **recommended for most cases**
  - `False`: 3D output `(batch_size, features, num_bits)` - Preserves feature structure

### Output Shape Examples

```python
import torch
from difflut.encoder import ThermometerEncoder

data = torch.randn(32, 100)  # 32 samples, 100 features

# With flatten=True (default)
encoder_flat = ThermometerEncoder(num_bits=4, flatten=True)
encoder_flat.fit(data)
output_flat = encoder_flat(data)
print(output_flat.shape)  # (32, 400) = 32 * 100 * 4 / 100 = batch * features * bits

# With flatten=False
encoder_3d = ThermometerEncoder(num_bits=4, flatten=False)
encoder_3d.fit(data)
output_3d = encoder_3d(data)
print(output_3d.shape)  # (32, 100, 4) = batch * features * bits
```

---

## Encoder Examples

### Thermometer Encoder (Recommended)

Most commonly used encoder. Maps each input value to a thermometer code.

```python
from difflut.encoder import ThermometerEncoder

# Create and fit
encoder = ThermometerEncoder(num_bits=8, flatten=True)
encoder.fit(train_data)

# Encode batches
x = torch.randn(32, 784)
encoded = encoder(x)
print(f"Input shape: {x.shape}")      # (32, 784)
print(f"Encoded shape: {encoded.shape}")  # (32, 6272) = 784 * 8

# The encoder learns:
# - min_val: minimum value across all features
# - max_val: maximum value across all features
# - Uses these to normalize and discretize all inputs
```

### Distributive Thermometer Encoder

Handles feature-specific value distributions (recommended for heterogeneous features):

```python
from difflut.encoder import DistributiveThermometerEncoder

encoder = DistributiveThermometerEncoder(num_bits=6, flatten=True)
encoder.fit(train_data)

# Learns per-feature min/max for better encoding
encoded = encoder(data)
```

### Gray Encoder

Minimizes Hamming distance between consecutive values:

```python
from difflut.encoder import GrayEncoder

# Useful when you care about smooth transitions
encoder = GrayEncoder(num_bits=8, flatten=True)
encoder.fit(train_data)
encoded = encoder(data)

# Gray code: Each increment changes only 1 bit
# 000 -> 001 -> 011 -> 010 -> 110 -> 111 -> 101 -> 100
```

### Binary Encoder

Standard binary representation (most compact):

```python
from difflut.encoder import BinaryEncoder

encoder = BinaryEncoder(num_bits=8, flatten=True)
encoder.fit(train_data)
encoded = encoder(data)

# Uses standard binary: 0 = 000, 1 = 001, 2 = 010, etc.
# Most compact but bit flips between neighbors can be large
```

### Gaussian Thermometer Encoder

Smooth encoding with Gaussian basis functions:

```python
from difflut.encoder import GaussianThermometerEncoder

encoder = GaussianThermometerEncoder(num_bits=8, flatten=True)
encoder.fit(train_data)
encoded = encoder(data)

# Smooth transitions, continuous-like behavior
# Good for regression or smooth function approximation
```

### One-Hot Encoder

Sparse one-hot representation:

```python
from difflut.encoder import OneHotEncoder

encoder = OneHotEncoder(num_bits=8, flatten=True)
encoder.fit(train_data)
encoded = encoder(data)

# Each value discretized into bins, one-hot across bins
# Very sparse representation
```

### Sign-Magnitude Encoder

For signed values:

```python
from difflut.encoder import SignMagnitudeEncoder

encoder = SignMagnitudeEncoder(num_bits=8, flatten=True)
encoder.fit(train_data)
encoded = encoder(data)

# Separate encoding for sign and magnitude
# Good for data with clear positive/negative distinction
```

### Logarithmic Encoder

For data with large value ranges:

```python
from difflut.encoder import LogarithmicEncoder

encoder = LogarithmicEncoder(num_bits=8, flatten=True)
encoder.fit(train_data)
encoded = encoder(data)

# Logarithmic scaling helps with exponential data
# Better for values spanning multiple orders of magnitude
```

---

## Fitting Best Practices

### Do's ✓

```python
from difflut.encoder import ThermometerEncoder

encoder = ThermometerEncoder(num_bits=8, flatten=True)

# ✓ Fit on training data
encoder.fit(train_data)
encoded_train = encoder(train_data)

# ✓ Use same fitted encoder for test data
encoded_test = encoder(test_data)

# ✓ Fit on representative sample for large datasets
sample = train_data[::10]  # Every 10th sample
encoder.fit(sample)

# ✓ Save and reuse fitted encoder
import torch
torch.save(encoder.state_dict(), 'encoder.pt')

# Load for inference
encoder_loaded = ThermometerEncoder(num_bits=8, flatten=True)
encoder_loaded.load_state_dict(torch.load('encoder.pt'))
encoded = encoder_loaded(new_data)

# ✓ Document encoder configuration with model
config = {
    'encoder_type': 'thermometer',
    'num_bits': 8,
    'flatten': True,
    'fitted_min': encoder.min_val.tolist(),
    'fitted_max': encoder.max_val.tolist()
}
```

### Don'ts ✗

```python
# ❌ Don't fit on test data (data leakage!)
encoder.fit(test_data)  # Wrong - leaks test info

# ❌ Don't use different encoders for train and test
encoder1 = ThermometerEncoder(num_bits=8)
encoder1.fit(train_data)
encoder2 = ThermometerEncoder(num_bits=8)
encoder2.fit(test_data)  # Wrong - inconsistent encoding

# ❌ Don't forget to fit before encoding
encoder = ThermometerEncoder(num_bits=8)
encoded = encoder(data)  # Wrong - encoder not fitted!

# ❌ Don't change num_bits or flatten after fitting
encoder = ThermometerEncoder(num_bits=8, flatten=True)
encoder.fit(data)
encoder.num_bits = 16  # Wrong - changes behavior
```

---

## Integration with Models

### With SimpleFeedForward Model

```python
from difflut.encoder import ThermometerEncoder
from difflut.models import ModelConfig, SimpleFeedForward
import torch

# Create encoder
encoder = ThermometerEncoder(num_bits=8, flatten=True)
encoder.fit(train_data)

# Create model
config = ModelConfig(
    model_type='feedforward',
    layer_type='random',
    node_type='probabilistic',
    encoder_config={
        'name': 'thermometer',
        'num_bits': 8,
        'flatten': True,
    },
    node_input_dim=6,
    layer_widths=[1024, 1000],
    num_classes=10,
    input_size=784,  # Raw input size before encoding
)

model = SimpleFeedForward(config)
model.fit_encoder(train_data)  # Fits encoder internally

# Forward pass
encoded = encoder(batch_data)
output = model(batch_data)  # Automatically encodes internally
```

### With Custom Layer Pipeline

```python
from difflut.encoder import ThermometerEncoder, DistributiveThermometerEncoder
from difflut.layers import RandomLayer
from difflut.nodes.node_config import NodeConfig
import torch
import torch.nn as nn

class CustomLUTModel(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        
        # Create and fit encoder
        self.encoder = ThermometerEncoder(num_bits=8, flatten=True)
        
        # Create layers
        node_config = NodeConfig(input_dim=6, output_dim=1)
        
        self.layer1 = RandomLayer(
            input_size=input_size * 8,  # After encoding
            output_size=512,
            node_type=LinearLUTNode,
            n=6,
            node_kwargs=node_config,
            seed=42
        )
        
        self.layer2 = RandomLayer(
            input_size=512,
            output_size=num_classes,
            node_type=LinearLUTNode,
            n=6,
            node_kwargs=node_config,
            seed=42
        )
    
    def fit_encoder(self, data):
        """Fit encoder on training data."""
        self.encoder.fit(data)
    
    def forward(self, x):
        x = x.flatten(1)  # Flatten input if needed
        x = self.encoder(x)  # Encode
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Usage
model = CustomLUTModel()
model.fit_encoder(train_data)
output = model(batch_data)
```

---

## Advanced: Custom Encoder Creation

See [Creating Components Guide](../../DEVELOPER_GUIDE/creating_components.md) for implementing custom encoders.

---

## Next Steps

- **[Layers Guide](layers.md)** - Build layer structures
- **[Nodes Guide](nodes.md)** - Understand node types and configuration
- **[Components Overview](components.md)** - High-level component reference
