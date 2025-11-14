# Creating Custom Components

Learn how to implement and register custom DiffLUT components. DiffLUT provides an extensible architecture for creating nodes, encoders, and layers.

---

## Quick Start

All components follow the same three-step pattern:

1. **Extend** the base class (`BaseNode`, `BaseEncoder`, or `BaseLUTLayer`)
2. **Register** using the `@register_*` decorator
3. **Implement** required methods

```python
from difflut import register_node
from difflut.nodes import BaseNode

@register_node('my_node')
class MyNode(BaseNode):
    def forward_train(self, x):
        # Training forward
        return x
    
    def forward_eval(self, x):
        # Eval forward
        return x
```

Then use it via the registry:

```python
from difflut.registry import REGISTRY

NodeClass = REGISTRY.get_node('my_node')
node = NodeClass(input_dim=6, output_dim=1)
```

---

## Component Guides

### [Creating Custom Nodes](creating_components/creating_nodes.md)
Complete guide for implementing LUT nodes with:
- Type-safe `NodeConfig` for structural parameters
- Custom initializers via `REGISTRY.get_initializer()`
- Custom regularizers via `REGISTRY.get_regularizer()`
- Forward train/eval methods
- Regularization terms
- Bitstream export for FPGA deployment

**Key Topics:**
- Basic node structure
- Initializers and regularizers
- Training vs. eval modes
- Gradient computation
- Testing nodes

**Examples:**
- Probabilistic nodes
- Learnable LUT nodes
- Polynomial approximation nodes

---

### [Creating Custom Layers](creating_components/creating_layers.md)
Complete guide for implementing connectivity layers with:
- Type-safe `LayerConfig` for training augmentation
- Random, learnable, and feature-wise routing
- Bit flipping during training
- Gradient stabilization (layerwise, nodewise, batch)
- Node composition and management

**Key Topics:**
- Layer connectivity patterns
- Training augmentation (bit flipping)
- Gradient stabilization techniques
- Forward pass routing
- Testing layers

**Examples:**
- Random routing layers
- Learnable routing layers
- Feature-wise grouping layers

---

### [Creating Custom Encoders](creating_components/creating_encoders.md)
Complete guide for implementing input encoders with:
- Type-safe encoder initialization
- Fitting to training data
- Forward encoding to binary representations
- Feature-wise vs. global quantization
- Device safety and edge cases

**Key Topics:**
- Encoder structure
- Fitting process
- Binary encoding methods
- Device transfers
- Testing encoders

**Examples:**
- Thermometer code encoders
- Adaptive resolution encoders
- Gray code encoders

---

## Type-Safe Configuration Overview

### NodeConfig

Use `NodeConfig` for type-safe node parameters:

```python
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY

# Retrieve functions from registry (not strings!)
init_fn = REGISTRY.get_initializer('kaiming_normal')
l2_reg = REGISTRY.get_regularizer('l2')

# Create config with actual function objects
node_config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=init_fn,           # Actual function, not string
    init_kwargs={'a': 0.0},    # Initializer arguments
    regularizers={'l2': l2_reg}  # Actual functions, not strings
)
```

**Critical Pattern:** Always retrieve from REGISTRY, never pass strings.

### LayerConfig

Use `LayerConfig` for training augmentation parameters:

```python
from difflut.layers.layer_config import LayerConfig

# Training augmentation config (not structural!)
layer_config = LayerConfig(
    flip_probability=0.05,           # Flip bits during training
    grad_stabilization='layerwise',  # Normalize gradients
    grad_target_std=1.0,             # Target gradient std
    grad_subtract_mean=False,        # Don't subtract mean
    grad_epsilon=1e-8,               # Numerical stability
)
```

**Separation:** LayerConfig handles training behavior, not architecture.

---

## Common Patterns

### 1. Module-Level Defaults

Always define defaults at module level in CAPITALS:

```python
DEFAULT_INPUT_DIM: int = 6
DEFAULT_OUTPUT_DIM: int = 1
DEFAULT_TEMPERATURE: float = 1.0

class MyNode(BaseNode):
    def __init__(self, input_dim=DEFAULT_INPUT_DIM, ...):
        # Use defaults
```

**Why:** Clear documentation of what's configurable, easy debugging.

### 2. Type Hints

Use full PEP 484 type hints:

```python
from typing import Optional, Dict, Any, Callable

def __init__(
    self,
    input_dim: int,
    output_dim: int,
    init_fn: Optional[Callable] = None,
    init_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
```

**Why:** IDE autocomplete, mypy checking, self-documenting code.

### 3. Docstrings

Use NumPy-style docstrings:

```python
def forward_train(self, x: torch.Tensor) -> torch.Tensor:
    """
    Training forward pass.
    
    Args:
        x: Input tensor of shape (batch_size, input_dim)
    
    Returns:
        Output tensor of shape (batch_size, output_dim)
    """
```

**Why:** Clear documentation, IDE tooltips.

### 4. Registry Retrieval

Always get components from registry:

```python
# ✓ CORRECT - Get from registry
init_fn = REGISTRY.get_initializer('kaiming_normal')
reg_fn = REGISTRY.get_regularizer('l2')

# ✗ WRONG - Don't pass strings
node_config = NodeConfig(init_fn='kaiming_normal')  # WRONG!
```

**Why:** Single source of truth, runtime validation, easy to test.

### 5. Device Safety

Move buffers to input device in forward:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    device = x.device
    
    # Move any buffers to device
    self.thresholds = self.thresholds.to(device)
    
    # Now compute...
```

**Why:** Works on any device (CPU, GPU, multi-GPU).

---

## Registration System

All components are registered automatically via decorators and stored in the global `REGISTRY`:

```python
from difflut.registry import REGISTRY

# Check registration
print(REGISTRY.list_nodes())      # ['probabilistic', 'dwn', 'linear_lut', ...]
print(REGISTRY.list_layers())     # ['random', 'learnable', ...]
print(REGISTRY.list_encoders())   # ['thermometer', 'distributive_thermometer', ...]
print(REGISTRY.list_initializers())  # ['kaiming_normal', 'xavier_normal', ...]
print(REGISTRY.list_regularizers())  # ['l2', 'l1', ...]

# Retrieve components
NodeClass = REGISTRY.get_node('probabilistic')
LayerClass = REGISTRY.get_layer('random')
EncoderClass = REGISTRY.get_encoder('thermometer')
init_fn = REGISTRY.get_initializer('kaiming_normal')
reg_fn = REGISTRY.get_regularizer('l2')

# Build models from registry
node = NodeClass(input_dim=6, output_dim=1)
layer = LayerClass(input_size=512, output_size=256, node_type='probabilistic', n=6)
encoder = EncoderClass(num_bits=8)
```

---

## Testing Your Components

Each component type has comprehensive test examples:

- **Nodes**: `tests/test_nodes/test_*.py`
- **Layers**: `tests/test_layers/test_*.py`
- **Encoders**: `tests/test_encoders/test_*.py`

See specific guides for complete testing patterns:

- [Node Testing](creating_components/creating_nodes.md#testing-your-node)
- [Layer Testing](creating_components/creating_layers.md#testing-your-layer)
- [Encoder Testing](creating_components/creating_encoders.md#testing-your-encoder)

---

## CUDA Support (Optional)

For performance-critical components, add CUDA kernels:

1. Write CUDA kernel: `difflut/<component>/cuda/my_kernel.cu`
2. Create Python wrapper: `difflut/<component>/cuda/my_kernel.py`
3. Update `setup.py` to compile
4. Add CPU fallback: use native PyTorch if CUDA unavailable

See specific guides for CUDA examples and patterns.

---

## Dimension Reference Table

Common dimensions used throughout DiffLUT:

| Parameter | Role | Typical Values | Notes |
|-----------|------|-----------------|-------|
| `input_dim` (node) | LUT input dimension (bits per node) | 4-8 | 2^n LUT entries |
| `output_dim` (node) | Output dimension per node | 1 | Usually scalar |
| `input_size` (layer/encoder) | Total input dimension | 64-2048 | After encoding |
| `output_size` (layer) | Number of nodes (output dim) | 100-1000 | Depends on task |
| `n` (layer parameter) | Bits routed to each node | 4-8 | Same as node input_dim |
| `num_bits` (encoder) | Quantization levels | 3-8 | More bits = more precision |
| `batch_size` | Samples per batch | 32-256 | Standard SGD batch size |

### Dimension Flow Example

```
Input: (batch=32, features=784)
  ↓ Encoder (thermometer, num_bits=4)
Encoded: (batch=32, features*num_bits=3136)
  ↓ Layer (input_size=3136, output_size=256, n=6)
  - Each of 256 nodes receives 6 random bits from 3136
  ↓ Layer output: (batch=32, 256)
```

---

## Next Steps

1. **Choose component type**: [Nodes](creating_components/creating_nodes.md), [Layers](creating_components/creating_layers.md), or [Encoders](creating_components/creating_encoders.md)
2. **Review examples** in respective guides
3. **Read base class** implementation (`difflut/<component>/base_*.py`)
4. **Implement** following patterns from guide
5. **Add tests** following test examples
6. **Integrate** via registry (automatic)
7. **Use in pipelines** with configuration files

---

## Resources

### Documentation
- [Nodes Guide](creating_components/creating_nodes.md)
- [Layers Guide](creating_components/creating_layers.md)
- [Encoders Guide](creating_components/creating_encoders.md)
- [Registry & Pipeline Guide](../USER_GUIDE/registry_pipeline.md)

### Code
- **Base Classes**: `difflut/nodes/base_node.py`, `difflut/layers/base_layer.py`, `difflut/encoder/base_encoder.py`
- **Config Classes**: `difflut/nodes/node_config.py`, `difflut/layers/layer_config.py`
- **Registry**: `difflut/registry.py`
- **Examples**: Existing implementations in respective directories

### Tests
- **Node Tests**: `tests/test_nodes/`
- **Layer Tests**: `tests/test_layers/`
- **Encoder Tests**: `tests/test_encoders/`

---

## FAQ

### Q: Should I use strings or function objects?
**A:** Always retrieve functions from REGISTRY: `REGISTRY.get_initializer('name')` returns function object. Pass the function object to config, not the string.

### Q: What if my component needs special initialization?
**A:** Use NodeConfig with `init_fn` and `init_kwargs`. The registry maintains all initializers, so create a new one if needed and register it with `@register_initializer`.

### Q: Can I register multiple versions of the same component?
**A:** Yes! Register with different names: `@register_node('my_node_v1')`, `@register_node('my_node_v2')`. Both will be available in REGISTRY.

### Q: How do I make my component GPU-compatible?
**A:** Move buffers to input device in forward: `buffer.to(x.device)`. Optional CUDA kernels for performance.

### Q: Where should I put my component code?
**A:** Create your component in appropriate directory (`difflut/nodes/`, `difflut/layers/`, `difflut/encoder/`) and add registration decorator.
