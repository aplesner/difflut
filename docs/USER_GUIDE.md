# User Guide

Welcome to the DiffLUT User Guide! This guide covers all the features you need to build and train LUT neural networks.

## Overview

DiffLUT is built around three core concepts:

1. **Encoders**: Transform continuous inputs into discrete representations suitable for LUT indexing

2. **Layers**: Connect inputs to nodes with specific connectivity patterns
3. **Nodes**: Individual LUT units that perform computation
3.1. **Initalizers**
3.2. **Regularizers**

Together, these form complete differentiable LUT networks.

## Quick Links

- **[Quick Start](QUICK_START.md)** - Get running in 5 minutes
- **[Components Guide](USER_GUIDE/components.md)** - Deep dive into encoders, nodes, and layers
- **[Registry & Pipelines](USER_GUIDE/registry_pipeline.md)** - Component discovery and configuration

## Architecture Overview

```
Input Data
    ↓
[Encoder] → Discretize continuous values → Binary/categorical indices
    ↓
[Layer] → Connect inputs to LUT nodes
    ↓
[Nodes] → Compute LUT values → Differentiable forward pass
    ↓
Output/Next Layer
```

## Component Categories

### Encoders
Transform continuous inputs into discrete representations. See [Components Guide](USER_GUIDE/components.md#encoders).

Available types: Thermometer, GaussianThermometer, DistributiveThermometer, Gray, OneHot, Binary, SignMagnitude, Logarithmic.

### Nodes
Define computation at individual LUT units. See [Components Guide](USER_GUIDE/components.md#nodes).

Available types: LinearLUT, PolyLUT, NeuralLUT, DWN, DWNStable, Probabilistic, Fourier, Hybrid.

### Layers
Connect inputs to nodes with configurable connectivity patterns. See [Components Guide](USER_GUIDE/components.md#layers).

Available types: Random, Learnable.

## Component Registry

DiffLUT's **component registry** provides a unified way to discover and instantiate components by name. This is especially useful when loading configurations from files.

See [Registry & Pipelines Guide](USER_GUIDE/registry_pipeline.md) for:
- Listing available components
- Dynamic component instantiation
- Configuration-driven model building
- Pipeline construction patterns

```python
from difflut.registry import get_node_class, get_registered_nodes

# List all available nodes
print(get_registered_nodes())

# Get node class by name
NodeClass = get_node_class('linear_lut')
node = NodeClass(input_dim=[4], output_dim=[1])
```

## Workflows

### Typical Training Workflow

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 1. Build model
model = MyLUTNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 2. Training loop
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        # Forward pass
        output = model(x_batch)
        loss = criterion(output, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    with torch.no_grad():
        val_loss = compute_validation_loss(model, val_loader)
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}")
```

### Configuration-Driven Model Building

Use the registry system to build models from configuration files:

```python
import yaml
from difflut.registry import get_layer_class, get_node_class, get_encoder_class

# Load config
with open('model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Build encoder
encoder_cls = get_encoder_class(config['encoder']['type'])
encoder = encoder_cls(**config['encoder']['params'])

# Build layers
layers = []
for layer_cfg in config['layers']:
    layer_cls = get_layer_class(layer_cfg['type'])
    node_cls = get_node_class(layer_cfg['node_type'])
    layer = layer_cls(
        input_size=layer_cfg['input_size'],
        output_size=layer_cfg['output_size'],
        node_type=node_cls,
        **layer_cfg.get('params', {})
    )
    layers.append(layer)
```

## Key Concepts

### Encoder Fitting

Encoders learn data statistics (min/max, quantiles) during fitting:

```python
from difflut.encoder import ThermometerEncoder

# Create encoder
encoder = ThermometerEncoder(num_bits=8)

# Fit to training data (learns value ranges)
encoder.fit(train_data)

# Now ready to use
encoded = encoder(test_data)
```

### Node Parameters

Each node type has different parameters. For example:

```python
from difflut.nodes import PolyLUTNode, NeuralLUTNode

# PolyLUT has degree parameter
poly_node = PolyLUTNode(
    input_dim=6,
    output_dim=1,
    degree=3  # Polynomial degree
)

# NeuralLUT has hidden width
neural_node = NeuralLUTNode(
    input_dim=4,
    output_dim=1,
    hidden_width=32  # MLP hidden layer width
)
```

**Note**: Nodes process 2D tensors with shape `(batch_size, input_dim)` → `(batch_size, output_dim)`. Layers use `nn.ModuleList` to manage multiple independent node instances.

### Layer Connectivity

Layers define how inputs connect to nodes. Each layer creates `output_size` independent nodes (stored in `nn.ModuleList`):

```python
from difflut.layers import RandomLayer, LearnableLayer
from difflut.nodes import LinearLUTNode

# Random: inputs randomly assigned to LUTs
# Creates output_size=256 independent LinearLUTNode instances
random_layer = RandomLayer(
    input_size=512,
    output_size=256,
    node_type=LinearLUTNode,
    n=4  # Each LUT gets 4 random inputs
)

# Learnable: learns which inputs each LUT uses
# Creates output_size=256 independent LinearLUTNode instances
learnable_layer = LearnableLayer(
    input_size=512,
    output_size=256,
    node_type=LinearLUTNode,
    n=4
)

# Forward pass processes (batch_size, 512) → (batch_size, 256)
# by iterating through the ModuleList of nodes
```

## Next Steps

1. **[Quick Start](QUICK_START.md)** - Build your first network
2. **[Components Guide](USER_GUIDE/components.md)** - Comprehensive component reference
3. **[Registry & Pipelines](USER_GUIDE/registry_pipeline.md)** - Advanced configuration patterns
4. **Examples** - Check `examples/` directory for full training examples

## Common Questions

**Q: Do I need to fit encoders?**
A: Yes, encoders learn value ranges from training data. Always call `encoder.fit(train_data)` before using.

**Q: Which node type should I use?**
A: Start with `LinearLUTNode` - it's fastest and often sufficient. Experiment with others if needed.

**Q: Can I mix different node types?**
A: Yes! Create different layers with different node types and stack them.

**Q: Is GPU support available?**
A: Yes! Fourier, Hybrid, and Gradient-Stabilized nodes have CUDA implementations for GPU acceleration.

**Q: How do I export models for FPGA?**
A: See the utils module for FPGA export tools. Check `difflut/utils/fpga_export.py`.

## Troubleshooting

**Encoder not fitted error**: Call `encoder.fit(data)` before using the encoder.

**Shape mismatch errors**: Check that layer input size matches encoder output size.

**GPU out of memory**: Reduce batch size or use smaller models.

**Slow training**: Check if using GPU-accelerated nodes for compute-heavy operations.

## Resources

- **Full Installation**: [Installation Guide](INSTALLATION.md)
- **Advanced Topics**: [Developer Guide](DEVELOPER_GUIDE.md)
- **Examples**: `examples/` directory
- **Source Code**: `difflut/` directory

