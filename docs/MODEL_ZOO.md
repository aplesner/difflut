"""
DiffLUT Model Zoo Documentation

# Model Zoo - Ready-to-Use Architectures

## Overview

The DiffLUT Model Zoo provides a curated collection of pre-designed neural network architectures
for common deep learning tasks. All models are optimized for DiffLUT's unique characteristics and
demonstrate different capabilities of the library.

**Goals:**
- Provide reference implementations for users to learn and build upon
- Demonstrate DiffLUT capabilities (encoding, bit flipping, gradient normalization, etc.)
- Enable quick benchmarking and comparison
- Show export-ready models for FPGA deployment

## Quick Start

```python
from difflut import get_model, list_models

# See all available models
print(list_models())

# Create a model
model = get_model('mnist_fc_8k_linear')

# Fit encoder on your data
model.fit_encoder(train_images)

# Use the model
predictions = model(test_images)
```

## Available Models

### MNIST (Fully Connected)

These are 2-layer fully connected models optimized for MNIST classification.
Designed for fast training and FPGA export.

#### MNISTLinearSmall (`mnist_fc_8k_linear`)

**Architecture:**
- Input: 28×28 MNIST images
- Encoder: Thermometer (4 bits) → 3136 features
- Layer 1: 3136 → 128 nodes (linear LUT, 4 inputs per node)
- Layer 2: 128 → 10 nodes
- Output: GroupSum classification

**Node type:** `linear_lut` (pure PyTorch LUT, no CUDA extensions)

**Best for:**
- Educational purposes
- Quick iteration and prototyping
- Platforms without CUDA

**Example:**
```python
from difflut import get_model

model = get_model('mnist_fc_8k_linear')
model.fit_encoder(train_images)
outputs = model(test_images)
```

#### MNISTDWNSmall (`mnist_fc_8k_dwn`)

**Architecture:** Same as MNISTLinearSmall

**Node type:** `dwn_stable` (CUDA-accelerated, FPGA-friendly)

**Best for:**
- FPGA export and deployment
- GPU acceleration
- Performance benchmarking
- Production models

**Example:**
```python
model = get_model('mnist_fc_8k_dwn')
model.fit_encoder(train_images)
```

### CIFAR-10 (Convolutional)

#### CIFAR10Conv (`cifar10_conv`)

**Architecture:**
- Input: 32×32 RGB images
- Encoder: Thermometer (4 bits) → 384 features
- Conv Layer 1: 384 channels → 128 features (kernel 3×3)
- Max Pooling: 2×2
- Conv Layer 2: 128 → 256 features (kernel 3×3)
- Max Pooling: 2×2
- FC Layer 1: Flattened → 256 nodes
- FC Layer 2: 256 → 10 classes
- Output: GroupSum classification

**Demonstrates:**
- Multi-channel input handling
- Convolutional LUT processing
- Spatial feature extraction

**Example:**
```python
model = get_model('cifar10_conv')
model.fit_encoder(train_images)  # (N, 3, 32, 32)
```

## Comparison Models

Models designed to showcase specific DiffLUT features and training techniques.

### Bit Flipping Variants

Demonstrate robustness to bit errors during inference (important for hardware deployment).

- `mnist_bitflip_none`: No bit flipping (baseline, 0%)
- `mnist_bitflip_5`: 5% bit flipping during training
- `mnist_bitflip_10`: 10% bit flipping during training
- `mnist_bitflip_20`: 20% bit flipping during training

**Use case:** Compare accuracy vs. hardware robustness tradeoff

```python
from difflut import get_model

models = {
    'baseline': get_model('mnist_bitflip_none'),
    'robust_5pct': get_model('mnist_bitflip_5'),
    'robust_10pct': get_model('mnist_bitflip_10'),
}

# Train each and compare accuracy
```

### Gradient Stabilization Variants

Demonstrate gradient normalization techniques for stable training.

- `mnist_gradnorm_none`: No stabilization (baseline)
- `mnist_gradnorm_layerwise`: Normalize per-layer gradients
- `mnist_gradnorm_batchwise`: Normalize per-batch gradients
- `mnist_gradnorm_layerwise_mean`: Normalize + mean subtraction

**Use case:** Study impact of gradient stabilization on convergence and final accuracy

```python
models = {
    'baseline': get_model('mnist_gradnorm_none'),
    'layerwise': get_model('mnist_gradnorm_layerwise'),
}
```

### Residual Initialization

#### MNISTResidualInit (`mnist_residual_init`)

Showcases residual layer initialization for improved learning in deep networks.

**Example:**
```python
model = get_model('mnist_residual_init')
```

## Model Architecture Details

### BaseModel

All models inherit from `BaseModel`, which provides:

- **Encoder Management**
  - `fit_encoder(data)`: Fit encoder on training data
  - Automatic handling of image flattening and encoding

- **Training Support**
  - `get_regularization_loss()`: Aggregate regularization from all layers
  - Support for custom loss functions

- **Utilities**
  - `count_parameters()`: Get parameter statistics
  - `get_layer_topology()`: Inspect network structure
  - `save_checkpoint()`, `load_checkpoint()`: Serialization

- **Device Handling**
  - Automatic device management
  - CPU/GPU transparency

### Common Parameters

Most models accept:
- `node_input_dim`: Number of inputs per LUT node (4 or 6)
- `encoder_bits`: Thermometer encoder bits (3-8)
- `flip_probability`: Bit flip probability [0, 1]
- `grad_stabilization`: Gradient stabilization mode
- `use_cuda`: Use CUDA extensions (if available)

## Workflow Examples

### Basic Training

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from difflut import get_model

# Load data
train_loader = DataLoader(...)
test_loader = DataLoader(...)

# Create model
model = get_model('mnist_fc_8k_linear')

# Fit encoder on training data
train_data = torch.cat([batch for batch, _ in train_loader])
model.fit_encoder(train_data)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch, targets in train_loader:
        optimizer.zero_grad()
        
        outputs = model(batch)
        loss = criterion(outputs, targets)
        
        # Add regularization
        reg_loss = model.get_regularization_loss()
        total_loss = loss + 0.001 * reg_loss
        
        total_loss.backward()
        optimizer.step()

# Evaluate
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch, targets in test_loader:
        outputs = model(batch)
        _, predicted = outputs.max(1)
        correct += (predicted == targets).sum()
        total += targets.size(0)

print(f"Accuracy: {100 * correct / total:.1f}%")
```

### Comparing Variants

```python
from difflut import list_models, get_model

# Get all bit flip variants
bitflip_models = [m for m in list_models() if 'bitflip' in m]

results = {}
for model_name in bitflip_models:
    model = get_model(model_name)
    # Train and evaluate
    results[model_name] = accuracy

# Analyze robustness vs accuracy tradeoff
```

### FPGA Export

```python
from difflut import get_model
from difflut.utils import export_to_c, export_to_fpga

# Use FPGA-friendly model
model = get_model('mnist_fc_8k_dwn')
model.fit_encoder(train_data)

# Export to C
export_to_c(model, 'model.h')

# Export to FPGA (HLS)
export_to_fpga(model, 'model.v')
```

## Implementation Guidelines

### For Model Zoo Developers

**Key principles:**
1. **Hardcoded Parameters**: All parameters hardcoded for reproducibility
2. **Self-Contained**: No external dependencies beyond difflut
3. **Documented**: Clear docstrings explaining design choices
4. **Testable**: Models should be instantiable without special setup

**Adding a New Model:**

1. Create a new model class inheriting from `BaseModel`
2. Implement `fit_encoder()`, `_build_layers()`, `_forward_impl()`
3. Add to `models/__init__.py` registry
4. Add tests in `tests/test_models/`
5. Document in this guide

Example:
```python
# in models/custom.py
from .base_model import BaseModel

class MyCustomModel(BaseModel):
    def __init__(self):
        super().__init__(name='my_model', input_size=784, num_classes=10)
        # Setup encoder, config, etc.
    
    def fit_encoder(self, data):
        # Fit encoder
        pass
    
    def _build_layers(self):
        # Build network
        pass
    
    def _forward_impl(self, x):
        # Forward pass
        pass
```

## Model Comparison

| Model | Nodes | Layers | Task | Export | Speed | Size |
|-------|-------|--------|------|--------|-------|------|
| MNISTLinearSmall | Linear LUT | 2 | MNIST | ✓ | Fast | Small |
| MNISTDWNSmall | DWN | 2 | MNIST | ✓ | GPU | Small |
| CIFAR10Conv | Linear LUT | Conv+FC | CIFAR-10 | ✓ | Medium | Medium |

## FAQ

### How do I customize a model?

Models support parameter overrides:
```python
model = get_model('mnist_fc_8k_linear', 
                  encoder_bits=6,
                  flip_probability=0.05)
```

### Can I use these for production?

Yes! Models are designed to be production-ready:
- Reproducible architectures
- Exportable to FPGA
- Tested and documented

### How do I contribute a new model?

1. Implement model inheriting from `BaseModel`
2. Add tests
3. Add documentation
4. Submit PR with examples

## References

- [User Guide](../USER_GUIDE.md)
- [Components Guide](../USER_GUIDE/components.md)
- [Registry & Pipelines](../USER_GUIDE/registry_pipeline.md)
- [Export Guide](../USER_GUIDE/export_guide.md)
"""

# Note: This is documentation text for display
# For programmatic use, see difflut.models module
