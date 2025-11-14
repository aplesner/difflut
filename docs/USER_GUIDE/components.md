# Components Guide

Overview of DiffLUT components. For detailed information, see the component-specific guides.

For creating custom components, see [Creating Components Guide](../DEVELOPER_GUIDE/creating_components.md).

---

## Component Overview

DiffLUT models are built from five main components:

```
Continuous Input
       |
   Encoder - Transform continuous input to discrete binary codes
       |
   Layers - Define connectivity between inputs and nodes
       |
   Nodes  - LUT-based computation units
       |
   Output
```

### Component Roles

Encoders transform continuous inputs to binary representations.

Layers define how inputs connect to nodes, supporting both fixed random and learnable routing patterns.

Nodes perform LUT-based computation on binary inputs, available in multiple types optimized for different use cases.

Initializers control parameter initialization for training stability.

Regularizers add constraints to improve generalization.

Blocks combine multiple layers into composite structures for specialized functions like convolution.

---

## Detailed Documentation

### Encoders

Transform continuous inputs to binary codes.

Learn about thermometer encoding, gray codes, binary encoding, and more.

[Read Encoders Guide](components/encoders.md)

---

### Layers

Define connectivity patterns between inputs and nodes.

Learn about random layers, learnable layers, and training augmentation options.

[Read Layers Guide](components/layers.md)

---

### Nodes

LUT-based computation units with initializers and regularizers.

Learn about LinearLUT, PolyLUT, NeuralLUT, DWN nodes, and more.

[Read Nodes Guide](components/nodes.md)

---

### Blocks

Composite multi-layer modules for specialized functions.

Learn about convolutional blocks and building custom composite structures.

[Read Blocks Guide](components/blocks.md)

---

## Dimension Reference

Typical dimensions through the DiffLUT pipeline:

| Stage | Input Shape | Output Shape | Notes |
|-------|-------------|--------------|-------|
| Raw Input | (batch, 784) | - | MNIST image |
| Encoded | (batch, 784) | (batch, 6272) | 784 features * 8 bits |
| Layer 1 | (batch, 6272) | (batch, 512) | First hidden layer |
| Layer 2 | (batch, 512) | (batch, 10) | Output layer |

---

## Component Selection

### Encoder Selection

Thermometer encoding: smooth, interpretable representations.

Binary encoding: most compact representation.

Gray codes: minimizes bit flip errors in adjacent values.

[See Encoders Guide for all types](components/encoders.md)

### Node Selection

LinearLUT: baseline, very fast and memory efficient.

PolyLUT: polynomial approximations, smooth functions.

NeuralLUT: complex mappings, slower but more expressive.

DWNStable: GPU-accelerated, memory efficient for large models.

[See Nodes Guide for all types](components/nodes.md)

### Layer Selection

RandomLayer: fixed random connectivity, deterministic.

LearnableLayer: trainable routing patterns, more expressive.

Use with LayerConfig for training augmentation options.

---

## Common Patterns

### Standard Classification

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

### Training with Augmentation

```python
config = ModelConfig(
    ...
    runtime={
        'flip_probability': 0.1,           # Bit flip augmentation
        'grad_stabilization': 'layerwise'  # Gradient stabilization
    }
)
```

### GPU-Accelerated Model

```python
config = ModelConfig(
    ...
    node_type='dwn_stable',  # GPU-friendly
)
```

### Device Placement (CPU/GPU)

DiffLUT models follow PyTorch conventions for device placement. Models always start on CPU and can be moved to GPU using standard PyTorch methods. The device automatically determines which kernels are used - no configuration needed.

```python
from difflut.models import ModelConfig, SimpleConvolutional

config = ModelConfig(
    model_type='convolutional',
    node_type='dwn_stable',
    ...
)

model = SimpleConvolutional(config)  # Always starts on CPU
output = model(input)  # Runs on CPU

model = model.cuda()  # Move to GPU
output = model(input)  # Now runs on GPU - CUDA kernels used automatically
```

You can also use `.to(device)` for more control:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
output = model(input.to(device))  # Runs on specified device

# Move back to CPU
model = model.cpu()
output = model(input)  # Runs on CPU again
```

**Key Points:**
- No `use_cuda` configuration needed - device placement is automatic
- CUDA kernels are used when model is on GPU (`.cuda()`)
- CPU kernels are used when model is on CPU (default)
- Move models and data to the same device for correct execution

---

## Documentation Links

Encoders: [Guide](components/encoders.md)

Layers: [Guide](components/layers.md)

Nodes: [Guide](components/nodes.md)

Blocks: [Guide](components/blocks.md)

Models: [Guide](components/models.md)

Registry & Pipelines: [Guide](registry_pipeline.md)

Creating Components: [Developer Guide](../DEVELOPER_GUIDE/creating_components.md)



