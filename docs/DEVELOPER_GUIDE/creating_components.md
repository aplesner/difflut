# Creating Custom Components

Learn how to implement and register custom DiffLUT components.

---

## Overview

DiffLUT provides an extensible architecture for creating custom nodes, encoders, layers, and blocks. All components follow the same three-step pattern: extend a base class, register with a decorator, and implement required methods.

---

## Quick Start

```python
from difflut import register_node
from difflut.nodes import BaseNode

@register_node('my_node')
class MyNode(BaseNode):
    def forward_train(self, x):
        return x
    
    def forward_eval(self, x):
        return x
```

Then use via the registry:

```python
from difflut.registry import REGISTRY

NodeClass = REGISTRY.get_node('my_node')
node = NodeClass(input_dim=6, output_dim=1)
```

---

## Component Types

### Nodes

LUT-based computation units with initializers and regularizers.

Implement `forward_train()`, `forward_eval()`, and optional `get_regularization_loss()`.

[Creating Custom Nodes](creating_components/creating_nodes.md)

### Layers

Connectivity patterns between inputs and nodes, supporting random and learnable routing.

Implement `forward()` and manage node composition.

[Creating Custom Layers](creating_components/creating_layers.md)

### Encoders

Transform continuous inputs to binary representations.

Implement `fit()` for training data adaptation and `forward()` for encoding.

[Creating Custom Encoders](creating_components/creating_encoders.md)

### Blocks

Composite multi-layer modules for specialized functions.

Implement `forward()` coordinating multiple layers, and optional `get_regularization_loss()`.

[Creating Custom Blocks](creating_components/creating_blocks.md)

---

## Configuration Pattern

All components use type-safe configuration objects:

```python
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY

# Retrieve from registry - never pass strings
init_fn = REGISTRY.get_initializer('kaiming_normal')
reg_fn = REGISTRY.get_regularizer('l2')

# Pass actual function objects to config
config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=init_fn,
    regularizers={'l2': reg_fn}
)
```

---

## Registration

Components are automatically discovered via decorators:

```python
from difflut import (
    register_node,
    register_layer,
    register_encoder,
    register_block,
    register_initializer,
    register_regularizer
)

# Each decorator automatically registers in REGISTRY
@register_node('my_node')
class MyNode: pass

@register_layer('my_layer')
class MyLayer: pass

@register_encoder('my_encoder')
class MyEncoder: pass

@register_block('my_block')
class MyBlock: pass

@register_initializer('my_init')
def my_initializer(tensor):
    pass

@register_regularizer('my_reg')
def my_regularizer(node):
    pass
```

Then discover:

```python
from difflut.registry import REGISTRY

REGISTRY.list_nodes()
REGISTRY.list_layers()
REGISTRY.list_encoders()
REGISTRY.list_blocks()
REGISTRY.list_initializers()
REGISTRY.list_regularizers()
```

---

## Type Hints and Documentation

Use full PEP 484 type hints and NumPy-style docstrings:

```python
from typing import Optional, Dict, Any, Callable
import torch.nn as nn

class MyComponent(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init_fn: Optional[Callable] = None,
    ) -> None:
        """
        Initialize custom component.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            init_fn: Initialization function from registry
        """
        super().__init__()
```

---

## Testing

Test forward pass, gradients, and device transfers:

```python
import torch
import pytest
from difflut.registry import REGISTRY

def test_my_component_forward():
    component = MyComponent(input_dim=6, output_dim=1)
    x = torch.randn(4, 6)
    output = component(x)
    assert output.shape == (4, 1)
    assert torch.isfinite(output).all()

def test_my_component_gradients():
    component = MyComponent(input_dim=6, output_dim=1)
    x = torch.randn(4, 6, requires_grad=True)
    output = component(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert (x.grad.abs() > 0).any()

def test_my_component_device():
    component = MyComponent(input_dim=6, output_dim=1)
    x = torch.randn(4, 6)
    
    # CPU
    out_cpu = component(x)
    assert out_cpu.device.type == 'cpu'
    
    # GPU (if available)
    if torch.cuda.is_available():
        component_gpu = component.cuda()
        x_gpu = x.cuda()
        out_gpu = component_gpu(x_gpu)
        assert out_gpu.device.type == 'cuda'
```

---

## Best Practices

1. Use `@register_*` decorators for automatic registration
2. Inherit from appropriate base class (BaseNode, BaseLUTLayer, BaseEncoder)
3. Use full type hints on all parameters and return values
4. Use NumPy-style docstrings on all public methods
5. Retrieve functions from REGISTRY - never pass strings
6. Support both train and eval modes where applicable
7. Test forward pass, gradients, and device transfers
8. Implement regularization loss collection if applicable
9. Define defaults at module level in CAPITALS
10. Move buffers to input device in forward for GPU compatibility

---

## Component Guides

Detailed implementation guides for each component type:

Creating Nodes: [Guide](creating_components/creating_nodes.md)

Creating Layers: [Guide](creating_components/creating_layers.md)

Creating Encoders: [Guide](creating_components/creating_encoders.md)

Creating Blocks: [Guide](creating_components/creating_blocks.md)

Creating Models: [Guide](creating_components/creating_models.md)

---

## Registry System

The global `REGISTRY` maintains all components:

```python
from difflut.registry import REGISTRY

# List registered components
nodes = REGISTRY.list_nodes()
layers = REGISTRY.list_layers()
encoders = REGISTRY.list_encoders()
blocks = REGISTRY.list_blocks()
initializers = REGISTRY.list_initializers()
regularizers = REGISTRY.list_regularizers()

# Retrieve for use
NodeClass = REGISTRY.get_node('my_node')
LayerClass = REGISTRY.get_layer('my_layer')
EncoderClass = REGISTRY.get_encoder('my_encoder')
BlockClass = REGISTRY.get_block('my_block')
init_fn = REGISTRY.get_initializer('my_initializer')
reg_fn = REGISTRY.get_regularizer('my_regularizer')
```

---

## Example Component

Simple node implementation showing the pattern:

```python
import torch
import torch.nn as nn
from difflut.nodes import BaseNode
from difflut import register_node

@register_node('example_node')
class ExampleNode(BaseNode):
    def __init__(self, input_dim: int = 6, output_dim: int = 1) -> None:
        super().__init__(input_dim, output_dim)
        self.linear = nn.Linear(2**input_dim, output_dim)
    
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        indices = (x * (2**self.input_dim - 1)).long()
        return self.linear(torch.nn.functional.one_hot(indices, 2**self.input_dim).float())
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_train(x)
    
    def get_regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)
```

---

## Dimension Reference

Common dimensions in DiffLUT:

| Parameter | Purpose | Typical Values |
|-----------|---------|-----------------|
| input_dim | Node LUT input (bits) | 4-8 |
| output_dim | Node output dimension | 1 |
| input_size | Layer/encoder input | 64-2048 |
| output_size | Layer output (nodes) | 100-1000 |
| n | Bits routed to each node | 4-8 |
| num_bits | Encoder quantization | 3-8 |
| batch_size | SGD batch size | 32-256 |

---

## Next Steps

1. Choose component type: [Nodes](creating_components/creating_nodes.md), [Layers](creating_components/creating_layers.md), [Encoders](creating_components/creating_encoders.md), or [Blocks](creating_components/creating_blocks.md)
2. Review examples in respective guides
3. Read base class implementation
4. Implement following patterns
5. Add comprehensive tests
6. Register automatically via decorator
7. Use in pipelines with configuration


