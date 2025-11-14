# Components Guide

High-level overview of DiffLUT components. For detailed information, see the component-specific guides.

> **For creating custom components**, see [Creating Components Guide](../DEVELOPER_GUIDE/creating_components.md).

---

## Table of Contents
1. [Component Overview](#component-overview)
2. [Detailed Documentation](#detailed-documentation)
3. [Dimension Reference](#dimension-reference)
4. [Quick Links](#quick-links)

---

## Component Overview

DiffLUT models are built from four main components that work together:

```
Continuous Input
       ↓
   [Encoder] → Discrete Binary Representation
       ↓
   [Layers] → Apply random or learned connections
       ↓
   [Nodes]  → LUT-based computation
       ↓
   Output
```

### Component Roles

| Component | Purpose | Details |
|-----------|---------|---------|
| **Encoder** | Transform continuous input to discrete binary codes | Map real values to indices for LUT lookup |
| **Layer** | Define connectivity pattern between inputs and nodes | Specify how nodes receive inputs (random or learnable) |
| **Node** | Compute output from binary inputs using LUT or learned function | Choose computation type (linear, polynomial, neural, etc.) |
| **Initializer** | Initialize node parameters (LUT weights) | Control training stability (Xavier, Kaiming, etc.) |
| **Regularizer** | Add constraints to node parameters | Improve generalization (L1, L2, spectral norm) |

### Typical Model Architecture

```python
import torch
import torch.nn as nn
from difflut.encoder import ThermometerEncoder
from difflut.layers import RandomLayer
from difflut.nodes import LinearLUTNode
from difflut.nodes.node_config import NodeConfig

class SimpleLUTModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Encoder: Continuous → Binary
        self.encoder = ThermometerEncoder(num_bits=8, flatten=True)
        
        # 2. Layers with Nodes: Binary → Output
        node_config = NodeConfig(input_dim=4, output_dim=1)
        
        self.layer1 = RandomLayer(
            input_size=6272,  # 784 features * 8 bits
            output_size=512,
            node_type=LinearLUTNode,
            n=4,
            node_kwargs=node_config
        )
        
        self.layer2 = RandomLayer(
            input_size=512,
            output_size=10,
            node_type=LinearLUTNode,
            n=4,
            node_kwargs=node_config
        )
    
    def fit_encoder(self, data):
        self.encoder.fit(data)
    
    def forward(self, x):
        x = x.flatten(1)
        x = self.encoder(x)      # [1] Encode
        x = self.layer1(x)       # [2] Layer with nodes
        x = self.layer2(x)       # [3] Final layer
        return x
```

---

## Detailed Documentation

Complete reference guides for each component:

### 📖 [Encoders Guide](components/encoders.md)

Transform continuous inputs to binary codes with adjustable resolution.

**Includes:**
- 8 available encoder types (Thermometer, Gray, Binary, Gaussian, etc.)
- Usage patterns and best practices
- Fitting strategies for different data sizes
- Complete integration examples

**Quick Example:**
```python
from difflut.encoder import ThermometerEncoder

encoder = ThermometerEncoder(num_bits=8, flatten=True)
encoder.fit(train_data)      # Learn data ranges
encoded = encoder(test_data) # Continuous → Binary
```

[👉 Read full Encoders Guide](components/encoders.md)

---

### 📖 [Layers Guide](components/layers.md)

Define connectivity between encoded inputs and nodes.

**Includes:**
- `LayerConfig` for type-safe training parameters
- `RandomLayer` for fixed random connectivity
- `LearnableLayer` for trainable connectivity
- Training configurations (bit flipping, gradient stabilization)
- 4 complete model architecture examples

**Quick Example:**
```python
from difflut.layers import RandomLayer
from difflut.nodes.node_config import NodeConfig

node_config = NodeConfig(input_dim=4, output_dim=1)
layer = RandomLayer(
    input_size=6272,
    output_size=512,
    node_type=LinearLUTNode,
    n=4,
    node_kwargs=node_config
)
output = layer(encoded_data)
```

[👉 Read full Layers Guide](components/layers.md)

---

### 📖 [Nodes Guide](components/nodes.md)

LUT-based computation units with initializers and regularizers.

**Includes:**
- `NodeConfig` for type-safe node configuration
- **8 Initializers** (Zeros, Ones, Uniform, Normal, Xavier, Kaiming, Variance Stabilized, Probabilistic)
- **3 Regularizers** (L1, L2, Spectral)
- **8 Node Types** (LinearLUT, PolyLUT, NeuralLUT, DWN, DWNStable, Probabilistic, Fourier, Hybrid)
- GPU acceleration patterns
- Complete training examples

**Quick Example:**
```python
from difflut.nodes import LinearLUTNode
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY

config = NodeConfig(
    input_dim=4,
    output_dim=1,
    init_fn=REGISTRY.get_initializer('kaiming_normal'),
    regularizers={'l2': REGISTRY.get_regularizer('l2')}
)
node = LinearLUTNode(**config.to_dict())
output = node(binary_input)
```

[👉 Read full Nodes Guide](components/nodes.md)

---

## Dimension Reference

Quick reference for tensor dimensions through the DiffLUT pipeline:

| Stage | Input Shape | Output Shape | Notes |
|-------|-------------|--------------|-------|
| **Raw Input** | `(batch, 784)` | - | MNIST image |
| **After Encoder** (flatten=True) | `(batch, 784)` | `(batch, 6272)` | 784 × 8 bits |
| **After Encoder** (flatten=False) | `(batch, 784)` | `(batch, 784, 8)` | Preserves structure |
| **After Layer 1** (512 nodes) | `(batch, 6272)` | `(batch, 512)` | RandomLayer output |
| **After Layer 2** (10 nodes) | `(batch, 512)` | `(batch, 10)` | Final classification |

### Dimension Calculation

```python
# Given:
batch_size = 32
input_features = 784        # MNIST
num_bits = 8               # Encoder bits
layer1_nodes = 512         # Layer 1 output
layer2_nodes = 10          # Layer 2 output (classes)

# Encoder output
encoded_size = input_features * num_bits  # 6272
encoded_shape = (batch_size, encoded_size)

# Layer outputs
layer1_output = (batch_size, layer1_nodes)  # (32, 512)
layer2_output = (batch_size, layer2_nodes)  # (32, 10)
```

### Architecture Note

- Layers use `nn.ModuleList` with `output_size` independent node instances
- Each node processes 2D tensors: `(batch, input_dim) → (batch, output_dim)`
- Layer output: `(batch_size, output_size * output_dim_per_node)`
- For typical case with `output_dim=1`: `(batch_size, output_size)`

---

## Quick Links

### Component Selection Guide

#### Encoder Selection

Choose encoder based on your data:

| Data Type | Recommended | Why |
|-----------|-------------|-----|
| General/Image | Thermometer | Smooth, interpretable |
| Audio/Signal | Gaussian Thermometer | Smooth transitions |
| Compact representation | Binary | Most compact |
| Minimizing bit flips | Gray | Smooth Hamming distance |

#### Node Selection

Choose node based on your needs:

| Use Case | Node Type | Speed | Memory | GPU |
|----------|-----------|-------|--------|-----|
| Baseline/Research | LinearLUT | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ |
| Smooth functions | PolyLUT | ⭐⭐⭐ | ⭐⭐⭐ | ❌ |
| Complex mappings | NeuralLUT | ⭐⭐ | ⭐⭐ | ❌ |
| Large models | DWNStable | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ |
| Uncertainty modeling | Probabilistic | ⭐⭐⭐ | ⭐⭐⭐ | ✅ |
| Periodic patterns | Fourier | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ |

#### Layer Selection

| Use Case | Recommendation |
|----------|---|
| Fixed random connectivity | `RandomLayer` |
| Learning connectivity | `LearnableLayer` |
| Need bit flip augmentation | Use `LayerConfig(flip_probability=0.1)` |
| Deep networks (5+ layers) | Use `LayerConfig(grad_stabilization='layerwise')` |

---

## Common Patterns

### Pattern 1: Standard MNIST Classification

```python
from difflut.models import ModelConfig, SimpleFeedForward

config = ModelConfig(
    model_type='feedforward',
    layer_type='random',
    node_type='linear_lut',
    encoder_config={'name': 'thermometer', 'num_bits': 8},
    layer_widths=[512, 256],
    num_classes=10,
    input_size=784
)

model = SimpleFeedForward(config)
model.fit_encoder(train_data)
output = model(test_data)
```

### Pattern 2: Robust Training with Augmentation

```python
config = ModelConfig(
    ...
    runtime={
        'flip_probability': 0.1,           # Bit flip augmentation
        'grad_stabilization': 'layerwise'  # Gradient stabilization
    }
)
```

### Pattern 3: GPU-Accelerated Model

```python
config = ModelConfig(
    ...
    node_type='dwn_stable',  # GPU-friendly node type
    runtime={'use_cuda': True}
)
```

### Pattern 4: Custom Initializer and Regularizer

```python
from difflut.registry import REGISTRY

config = ModelConfig(
    ...,
    node_config={
        'init_fn': REGISTRY.get_initializer('kaiming_normal'),
        'regularizers': {
            'l2': REGISTRY.get_regularizer('l2')
        }
    }
)
```

---

## Related Documentation

- **[Registry & Pipeline Guide](registry_pipeline.md)** - Dynamic component discovery and configuration-driven model building
- **[DEVELOPER_GUIDE](../DEVELOPER_GUIDE/creating_components.md)** - Create custom encoders, nodes, or layers
- **[QUICK_START](../QUICK_START.md)** - Get started with a simple example
- **[API_REFERENCE](../../API_REFERENCE.md)** - Complete API documentation

---

## Next Steps

1. **Choose your starting point:**
   - 🏃 Want to build a model quickly? Start with [Quick Start](../QUICK_START.md)
   - 🔧 Need encoder details? Read [Encoders Guide](components/encoders.md)
   - 🧩 Building custom architecture? Read [Layers Guide](components/layers.md)
   - 🎯 Tuning node behavior? Read [Nodes Guide](components/nodes.md)
   - 📦 Dynamic model building? See [Registry & Pipelines](registry_pipeline.md)

2. **Check the specific guide for your component:**
   - Each guide includes complete examples and best practices
   - Learn which components work best for your use case
   - Understand parameter tuning for performance

3. **Explore advanced topics:**
   - GPU acceleration with CUDA-enabled nodes
   - Custom initializers and regularizers
   - Building pipelines for hyperparameter search

