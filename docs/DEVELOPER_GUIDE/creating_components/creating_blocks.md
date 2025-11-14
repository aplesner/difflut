# Creating Custom Blocks

Learn how to implement custom DiffLUT blocks for composite multi-layer modules.

---

## Overview

Blocks are composite modules combining multiple layers to implement specialized functions. Unlike individual layers (which map inputs to LUT nodes), blocks orchestrate complete subsystems.

Use blocks when you need:
- Multi-layer hierarchies (tree structures)
- Specialized processing patterns (convolution, pooling, etc.)
- Coordinated feature processing

---

## Quick Start

Create a simple block:

```python
import torch
import torch.nn as nn
from difflut.blocks import ConvolutionalLayer
from difflut.layers import LayerConfig
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY
from difflut import register_block

@register_block('my_block')
class MyBlock(nn.Module):
    def __init__(self, config, node_type, node_kwargs, layer_type, layer_config):
        super().__init__()
        self.config = config
        self.node_type = node_type
        self.layer_type = layer_type
        
        # Build internal structure
        self.layers = nn.ModuleList()
        self._build()
    
    def _build(self):
        """Build internal layer structure."""
        pass
    
    def forward(self, x):
        """Forward pass."""
        return x
```

Then use it:

```python
from difflut.registry import REGISTRY

BlockClass = REGISTRY.get_block('my_block')
block = BlockClass(config, node_type, node_kwargs, layer_type, layer_config)
```

---

## Base Classes

Blocks inherit from `nn.Module` and follow the standard PyTorch pattern:

```python
import torch.nn as nn
from difflut import register_block

@register_block('block_name')
class MyBlock(nn.Module):
    def __init__(self, config, node_type, node_kwargs, layer_type, layer_config):
        super().__init__()
        self.config = config
        self.node_type = node_type
        self.layer_type = layer_type
        
    def forward(self, x):
        """Implement forward pass."""
        raise NotImplementedError
    
    def get_regularization_loss(self):
        """Optional: Return regularization loss."""
        return 0.0
```

---

## Configuration

Blocks use configuration objects for type-safe parameter management:

```python
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class MyBlockConfig:
    """Configuration for custom block."""
    param1: int
    param2: float
    param3: Dict[str, Any]
```

---

## Example: Tree-Based Block

```python
import torch
import torch.nn as nn
from difflut.layers import RandomLayer
from difflut import register_block

@register_block('tree_block')
class TreeBlock(nn.Module):
    """Tree-structured multi-layer block."""
    
    def __init__(self, config, node_type, node_kwargs, layer_type, layer_config):
        super().__init__()
        self.config = config
        self.node_type = node_type
        self.layer_type = layer_type
        
        self.layers = nn.ModuleList()
        self._build_tree(config.depth, config.input_size, config.output_size)
    
    def _build_tree(self, depth, input_size, output_size):
        """Build tree hierarchy."""
        if depth == 0:
            # Leaf layer
            layer = self.layer_type(
                input_size=input_size,
                output_size=output_size,
                node_type=self.node_type,
                node_kwargs=self.node_kwargs,
                n=6,
            )
            self.layers.append(layer)
        else:
            # Internal layer
            hidden_size = (input_size + output_size) // 2
            layer = self.layer_type(
                input_size=input_size,
                output_size=hidden_size,
                node_type=self.node_type,
                node_kwargs=self.node_kwargs,
                n=6,
            )
            self.layers.append(layer)
            
            # Recurse
            self._build_tree(depth - 1, hidden_size, output_size)
    
    def forward(self, x):
        """Forward through tree."""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_regularization_loss(self):
        """Collect regularization from all layers."""
        loss = 0.0
        for layer in self.layers:
            if hasattr(layer, 'get_regularization_loss'):
                loss = loss + layer.get_regularization_loss()
        return loss
```

---

## Common Patterns

### Pattern 1: Sequential Block

```python
class SequentialBlock(nn.Module):
    def __init__(self, config, node_type, node_kwargs, layer_type, layer_config):
        super().__init__()
        self.layers = nn.ModuleList()
        
        current_size = config.input_size
        for output_size in config.layer_widths:
            layer = layer_type(
                input_size=current_size,
                output_size=output_size,
                node_type=node_type,
                node_kwargs=node_kwargs,
                layer_config=layer_config,
            )
            self.layers.append(layer)
            current_size = output_size
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

### Pattern 2: Residual Block

```python
class ResidualBlock(nn.Module):
    def __init__(self, config, node_type, node_kwargs, layer_type, layer_config):
        super().__init__()
        
        self.main = layer_type(
            input_size=config.size,
            output_size=config.size,
            node_type=node_type,
            node_kwargs=node_kwargs,
            layer_config=layer_config,
        )
    
    def forward(self, x):
        return x + self.main(x)  # Residual connection
```

### Pattern 3: Branched Block

```python
class BranchedBlock(nn.Module):
    def __init__(self, config, node_type, node_kwargs, layer_type, layer_config):
        super().__init__()
        
        self.branch1 = layer_type(
            input_size=config.input_size,
            output_size=config.branch_size,
            node_type=node_type,
            node_kwargs=node_kwargs,
            layer_config=layer_config,
        )
        
        self.branch2 = layer_type(
            input_size=config.input_size,
            output_size=config.branch_size,
            node_type=node_type,
            node_kwargs=node_kwargs,
            layer_config=layer_config,
        )
    
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        return torch.cat([out1, out2], dim=1)
```

---

## Type Hints and Documentation

Use full type hints and NumPy-style docstrings:

```python
from typing import Optional, Dict, Any, Callable
import torch

class MyBlock(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
        node_type: Callable,
        node_kwargs: Dict[str, Any],
        layer_type: Callable,
        layer_config: Optional[Any] = None,
    ) -> None:
        """
        Initialize custom block.
        
        Args:
            config: Block configuration dictionary
            node_type: Node class from registry
            node_kwargs: Node initialization arguments
            layer_type: Layer class from registry
            layer_config: Optional layer configuration for training augmentation
        """
        super().__init__()
```

---

## Regularization

Implement regularization loss collection:

```python
def get_regularization_loss(self) -> torch.Tensor:
    """
    Compute total regularization loss.
    
    Returns:
        Scalar tensor with regularization loss
    """
    loss = torch.tensor(0.0, device=self.device)
    for layer in self.layers:
        if hasattr(layer, 'get_regularization_loss'):
            loss = loss + layer.get_regularization_loss()
    return loss
```

---

## Testing

Example tests for custom blocks:

```python
import torch
import pytest
from difflut.registry import REGISTRY
from difflut.nodes.node_config import NodeConfig
from difflut.layers import LayerConfig

def test_my_block_forward():
    """Test forward pass shape."""
    config = {...}
    node_type = REGISTRY.get_node('probabilistic')
    layer_type = REGISTRY.get_layer('random')
    node_config = NodeConfig(input_dim=6, output_dim=1)
    
    block = MyBlock(config, node_type, node_config, layer_type, LayerConfig())
    
    x = torch.randn(8, config.input_size)
    output = block(x)
    
    assert output.shape == (8, config.output_size)
    assert torch.isfinite(output).all()

def test_my_block_gradients():
    """Test gradient flow."""
    config = {...}
    node_type = REGISTRY.get_node('probabilistic')
    layer_type = REGISTRY.get_layer('random')
    node_config = NodeConfig(input_dim=6, output_dim=1)
    
    block = MyBlock(config, node_type, node_config, layer_type, LayerConfig())
    block.train()
    
    x = torch.randn(8, config.input_size, requires_grad=True)
    output = block(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None
    assert (x.grad.abs() > 0).any()

def test_my_block_regularization():
    """Test regularization loss."""
    config = {...}
    node_type = REGISTRY.get_node('probabilistic')
    layer_type = REGISTRY.get_layer('random')
    node_config = NodeConfig(input_dim=6, output_dim=1)
    
    block = MyBlock(config, node_type, node_config, layer_type, LayerConfig())
    
    reg_loss = block.get_regularization_loss()
    assert reg_loss >= 0.0
```

---

## Registration

Register your block:

```python
from difflut import register_block

@register_block('my_block_name')
class MyBlock(nn.Module):
    pass
```

Then use via registry:

```python
from difflut.registry import REGISTRY

BlockClass = REGISTRY.get_block('my_block_name')
block = BlockClass(config, node_type, node_config, layer_type, layer_config)
```

---

## Best Practices

1. Use `@register_block` decorator for automatic registration
2. Inherit from `nn.Module` for PyTorch compatibility
3. Use type hints for all parameters and return values
4. Include NumPy-style docstrings on all methods
5. Implement `get_regularization_loss()` for training
6. Support both train and eval modes
7. Test forward pass, gradients, and device transfers
8. Document configuration parameters clearly

---

## Next Steps

- See [Blocks Guide](../../USER_GUIDE/components/blocks.md) for usage examples
- See [Layers Guide](../../DEVELOPER_GUIDE/creating_components/creating_layers.md) for underlying layer implementation
- See [Registry & Pipelines](../../USER_GUIDE/registry_pipeline.md) for component registration system

