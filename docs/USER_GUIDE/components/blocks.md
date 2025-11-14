# Blocks Guide

Learn how to use DiffLUT blocks for building composite multi-layer modules.

---

## Overview

Blocks are composite modules that combine multiple layers to perform specialized functions. Unlike individual layers which connect inputs to single LUT nodes, blocks orchestrate multiple layers to implement complete subsystems like convolutional layers.

Currently available blocks:

- **ConvolutionalLayer**: Tree-based convolutional processing using LUT-based nodes

---

## ConvolutionalLayer

Implements efficient convolution using a tree of LUT-based layers for processing convolutional patches.

### Basic Usage

```python
from difflut.blocks import ConvolutionalLayer, ConvolutionConfig
from difflut.layers import LayerConfig
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY

# Configuration
conv_config = ConvolutionConfig(
    tree_depth=1,
    in_channels=3,
    out_channels=16,
    receptive_field=3,
    stride=1,
    padding=0,
    chunk_size=64,
    seed=42,
)

layer_config = LayerConfig()
node_config = NodeConfig(input_dim=6, output_dim=1)

node_type = REGISTRY.get_node("probabilistic")
layer_type = REGISTRY.get_layer("random")

# Create convolutional layer
conv = ConvolutionalLayer(
    convolution_config=conv_config,
    node_type=node_type,
    node_kwargs=node_config,
    layer_type=layer_type,
    n_inputs_per_node=6,
    layer_config=layer_config,
)

# Use
input_tensor = torch.randn(4, 3, 16, 16)  # (batch, channels, height, width)
output = conv(input_tensor)  # (batch, 16, 14, 14)
```

### ConvolutionConfig

Type-safe configuration for convolutional blocks:

```python
from difflut.blocks import ConvolutionConfig

config = ConvolutionConfig(
    # Structural parameters
    tree_depth=1,            # Depth of tree hierarchy
    in_channels=3,           # Input channels
    out_channels=16,         # Output channels
    receptive_field=3,       # Kernel size (always square)
    stride=1,                # Stride for convolution
    padding=0,               # Padding mode
    chunk_size=64,           # Batch size for chunked processing
    seed=42,                 # Random seed for reproducibility
)
```

### Output Shape Calculation

For a convolutional layer without padding:

```
output_height = (input_height - receptive_field) / stride + 1
output_width = (input_width - receptive_field) / stride + 1
```

Example: 16x16 input with receptive_field=3, stride=1:
- Output: (16-3)/1 + 1 = 14x14

### Grouped Connections

For complex inputs, use grouped connections to ensure full channel coverage:

```python
conv = ConvolutionalLayer(
    convolution_config=conv_config,
    node_type=node_type,
    node_kwargs=node_config,
    layer_type=layer_type,
    n_inputs_per_node=6,
    layer_config=layer_config,
    grouped_connections=True,      # Ensure full coverage
    ensure_full_coverage=True,      # Check all channels used
)
```

---

## Common Patterns

### Pattern 1: Image Processing Pipeline

```python
import torch
from difflut.blocks import ConvolutionalLayer, ConvolutionConfig
from difflut.layers import LayerConfig, RandomLayer
from difflut.registry import REGISTRY

# Convolutional feature extraction
conv_config = ConvolutionConfig(
    tree_depth=1,
    in_channels=3,
    out_channels=32,
    receptive_field=3,
    stride=1,
    padding=0,
    chunk_size=128,
    seed=42,
)

node_type = REGISTRY.get_node("probabilistic")
layer_type = REGISTRY.get_layer("random")
node_config = NodeConfig(input_dim=6, output_dim=1)

conv = ConvolutionalLayer(
    convolution_config=conv_config,
    node_type=node_type,
    node_kwargs=node_config,
    layer_type=layer_type,
    n_inputs_per_node=6,
    layer_config=LayerConfig(),
)

# Dense layers for classification
dense = RandomLayer(
    input_size=32 * 14 * 14,  # Flattened conv output
    output_size=128,
    node_type=node_type,
    node_kwargs=node_config,
    seed=43,
)

# Forward pass
x = torch.randn(8, 3, 16, 16)
x = conv(x)              # (8, 32, 14, 14)
x = x.reshape(8, -1)    # (8, 6272)
x = dense(x)            # (8, 128)
```

### Pattern 2: Multi-Scale Processing

```python
# Process at different scales
scales = [
    ConvolutionConfig(
        tree_depth=1,
        in_channels=3,
        out_channels=16,
        receptive_field=3,
        stride=1,
        padding=0,
        chunk_size=64,
        seed=42 + i,
    )
    for i in range(3)
]

convs = [
    ConvolutionalLayer(
        convolution_config=cfg,
        node_type=node_type,
        node_kwargs=node_config,
        layer_type=layer_type,
        n_inputs_per_node=6,
        layer_config=LayerConfig(),
    )
    for cfg in scales
]

# Process with each scale
x = torch.randn(8, 3, 16, 16)
outputs = [conv(x) for conv in convs]  # Multiple scales
combined = torch.cat(outputs, dim=1)  # Combine features
```

---

## Configuration Parameters

### ConvolutionConfig Parameters

| Parameter | Type | Purpose |
|-----------|------|---------|
| tree_depth | int | Hierarchy depth for layer organization |
| in_channels | int | Number of input channels |
| out_channels | int | Number of output channels |
| receptive_field | int | Kernel size (square) |
| stride | int | Convolution stride |
| padding | int | Padding size |
| chunk_size | int | Batch processing chunk size |
| seed | int | Random seed for reproducibility |

### LayerConfig Parameters (for training)

| Parameter | Type | Purpose |
|-----------|------|---------|
| flip_probability | float | Probability of bit flipping during training |
| grad_stabilization | str | Gradient normalization: "none", "layerwise", "nodewise", "batch" |
| grad_target_std | float | Target standard deviation for gradients |
| grad_subtract_mean | bool | Whether to subtract mean from gradients |
| grad_epsilon | float | Numerical stability constant |

---

## Testing Blocks

Example test for convolutional blocks:

```python
import torch
import pytest
from difflut.blocks import ConvolutionalLayer, ConvolutionConfig
from difflut.layers import LayerConfig
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY

def test_convolutional_forward():
    config = ConvolutionConfig(
        tree_depth=1,
        in_channels=3,
        out_channels=8,
        receptive_field=3,
        stride=1,
        padding=0,
        chunk_size=32,
        seed=42,
    )
    
    node_type = REGISTRY.get_node("probabilistic")
    layer_type = REGISTRY.get_layer("random")
    node_config = NodeConfig(input_dim=6, output_dim=1)
    
    conv = ConvolutionalLayer(
        convolution_config=config,
        node_type=node_type,
        node_kwargs=node_config,
        layer_type=layer_type,
        n_inputs_per_node=6,
        layer_config=LayerConfig(),
    )
    
    x = torch.randn(4, 3, 16, 16)
    output = conv(x)
    
    assert output.shape == (4, 8, 14, 14)
    assert torch.isfinite(output).all()
```

---

## Next Steps

- See [Creating Custom Blocks](../../DEVELOPER_GUIDE/creating_components/creating_blocks.md) to build custom blocks
- See [Layers Guide](layers.md) for underlying layer implementations
- See [Models Guide](models.md) for using blocks in complete models

