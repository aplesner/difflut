# Components Guide

A comprehensive reference for using DiffLUT components: encoders, layers, nodes, and utility modules.

> **For creating custom components**, see [Creating Components Guide](../DEVELOPER_GUIDE/creating_components.md).

---

## Table of Contents
1. [Dimension Reference](#dimension-reference)
2. [Encoders](#encoders)
3. [Layers](#layers)
4. [Nodes](#nodes)
5. [Modules (GroupSum)](#modules)

---

## Dimension Reference

Quick reference for tensor dimensions through the DiffLUT pipeline:

| Component | Input Shape | Output Shape | Notes |
|-----------|-------------|--------------|-------|
| **Encoder** (flatten=True) | `(batch_size, input_dim)` | `(batch_size, input_dim * num_bits)` | Default: auto-flattens to 2D |
| **Encoder** (flatten=False) | `(batch_size, input_dim)` | `(batch_size, input_dim, num_bits)` | Preserves 3D structure |
| **Layer** | `(batch_size, input_size)` | `(batch_size, output_size * node_output_dim)` | Routes to `output_size` independent nodes |
| **Single Node** | `(batch_size, node_input_dim)` | `(batch_size, node_output_dim)` | Each node processes 2D tensors independently |
| **GroupSum** | `(batch_size, num_features)` | `(batch_size, k)` | Groups features and sums within groups |

**Architecture Note**: Layers use `nn.ModuleList` with `output_size` independent node instances. Each node processes 2D tensors `(batch, input_dim) → (batch, output_dim)`. Layers process nodes efficiently using preallocated output tensors.

---



## Encoders

Encoders transform continuous input values into discrete binary representations suitable for LUT indexing. All encoders inherit from `BaseEncoder` and provide a consistent API.

### Basic Usage Pattern

```python
from difflut.encoder import ThermometerEncoder
import torch

# 1. Create encoder
encoder = ThermometerEncoder(num_bits=8, flatten=True)

# 2. Fit to data (learns min/max ranges)
train_data = torch.randn(1000, 784)
encoder.fit(train_data)

# 3. Encode inputs
x = torch.randn(32, 784)  # (batch_size, features)
encoded = encoder(x)       # (batch_size, features * num_bits) if flatten=True

# 4. Use in model
layer = RandomLayer(input_size=encoded.shape[1], ...)
output = layer(encoded)
```

### Common Parameters

All encoders support:
- `num_bits`: Number of bits per feature (resolution)
- `flatten`: If `True` (default), outputs 2D `(batch, features * num_bits)`; if `False`, outputs 3D `(batch, features, num_bits)`

### Available Encoders

| Encoder | Description | Best For |
|---------|-------------|----------|
| `ThermometerEncoder` | Unary encoding (threshold-based) | Smooth, interpretable discretization |
| `GrayEncoder` | Gray code (minimal Hamming distance) | Minimizing bit flips between neighbors |
| `BinaryEncoder` | Standard binary encoding | Compact representation |
| `GaussianThermometerEncoder` | Gaussian basis functions | Smooth transitions with continuous-like encoding |
| `DistributiveThermometerEncoder` | Distributive thermometer | Handling value distributions |
| `OneHotEncoder` | One-hot encoding | Sparse, interpretable bins |
| `SignMagnitudeEncoder` | Sign-magnitude representation | Signed values with clear positive/negative |
| `LogarithmicEncoder` | Logarithmic scaling | Large value ranges, exponential data |

### Example: Thermometer Encoder

```python
from difflut.encoder import ThermometerEncoder

encoder = ThermometerEncoder(num_bits=8, flatten=True)
encoder.fit(train_data)

# Input: (32, 784) continuous values
# Output: (32, 6272) binary values (784 * 8)
encoded = encoder(data)
```

### Example: Gray Encoder

```python
from difflut.encoder import GrayEncoder

# Minimizes Hamming distance between consecutive values
encoder = GrayEncoder(num_bits=8, flatten=True)
encoder.fit(train_data)
encoded = encoder(data)
```

### Fitting Best Practices

```python
# ✓ Fit on training data
encoder.fit(train_data)

# ✓ Fit on representative sample for large datasets
sample = train_data[::10]  # Every 10th sample
encoder.fit(sample)

# ❌ Don't fit on test data (data leakage)
encoder.fit(test_data)  # Wrong!

# ✓ Save and reuse fitted encoder
torch.save(encoder.state_dict(), 'encoder.pt')
encoder.load_state_dict(torch.load('encoder.pt'))
```

---

---

## Layers

Layers define connectivity patterns between encoded inputs and LUT nodes. DiffLUT provides two main layer types: `RandomLayer` (fixed random connectivity) and `LearnableLayer` (trainable connectivity).

### LayerConfig: Type-Safe Configuration

Use `LayerConfig` for type-safe layer training parameters:

```python
from difflut.layers.layer_config import LayerConfig

# Create configuration for training augmentation
layer_config = LayerConfig(
    flip_probability=0.1,           # Flip 10% of bits during training
    grad_stabilization='layerwise', # Normalize gradients per layer
    grad_target_std=1.0,            # Target standard deviation
    grad_subtract_mean=False,       # Don't center gradients
    grad_epsilon=1e-8               # Numerical stability
)
```

**LayerConfig Parameters:**
- `flip_probability`: Bit flip augmentation probability [0, 1] (default: 0.0)
- `grad_stabilization`: Gradient normalization mode: `'none'`, `'layerwise'`, `'batchwise'` (default: `'none'`)
- `grad_target_std`: Target standard deviation for gradient rescaling (default: 1.0)
- `grad_subtract_mean`: Whether to center gradients before rescaling (default: False)
- `grad_epsilon`: Small constant for numerical stability (default: 1e-8)

### RandomLayer: Fixed Random Connectivity

Creates fixed random connections between inputs and nodes. Each input is used at least once per node before reuse.

```python
from difflut.layers import RandomLayer
from difflut.nodes import LinearLUTNode
from difflut.nodes.node_config import NodeConfig
from difflut.layers.layer_config import LayerConfig

# Node configuration
node_config = NodeConfig(
    input_dim=4,
    output_dim=1
)

# Layer training configuration (optional)
layer_config = LayerConfig(
    flip_probability=0.1,
    grad_stabilization='layerwise'
)

# Create layer
layer = RandomLayer(
    input_size=6272,           # Encoded input features
    output_size=128,           # Number of nodes
    node_type=LinearLUTNode,
    n=4,                       # Inputs per node
    node_kwargs=node_config,
    layer_config=layer_config, # Optional training config
    seed=42                    # Random seed for reproducibility
)

# Forward pass
x = torch.randn(32, 6272)
output = layer(x)  # (32, 128) if output_dim=1
```

**Parameters:**
- `input_size`: Number of input features (from encoder or previous layer)
- `output_size`: Number of LUT nodes in the layer
- `node_type`: Node class (e.g., `LinearLUTNode`, `DWNNode`)
- `n`: Number of inputs per node (must match `node_config.input_dim`)
- `node_kwargs`: `NodeConfig` instance with node parameters
- `layer_config`: `LayerConfig` instance for training parameters (optional)
- `seed`: Random seed for reproducible mapping

### LearnableLayer: Trainable Connectivity

Learns optimal connections between inputs and nodes during training.

```python
from difflut.layers import LearnableLayer

layer = LearnableLayer(
    input_size=6272,
    output_size=128,
    node_type=LinearLUTNode,
    n=4,
    node_kwargs=node_config,
    layer_config=layer_config,  # Optional
    temperature=1.0,            # Gumbel-Softmax temperature
    hard=False                  # Use soft sampling during training
)

# Forward pass
output = layer(x)
```

**Additional Parameters:**
- `temperature`: Gumbel-Softmax temperature for connectivity sampling (default: 1.0)
- `hard`: If True, use hard one-hot sampling; if False, use soft differentiable sampling (default: False)

### Layer Usage Patterns

```python
# Standard usage with defaults (no augmentation)
layer = RandomLayer(
    input_size=100,
    output_size=50,
    node_type=LinearLUTNode,
    n=4,
    node_kwargs=node_config
)

# Robust training with bit flipping
robust_config = LayerConfig(flip_probability=0.1)
layer = RandomLayer(
    input_size=100,
    output_size=50,
    node_type=LinearLUTNode,
    n=4,
    node_kwargs=node_config,
    layer_config=robust_config
)

# Gradient stabilization for deep networks
stable_config = LayerConfig(
    grad_stabilization='layerwise',
    grad_target_std=1.0
)
layer = RandomLayer(
    input_size=100,
    output_size=50,
    node_type=LinearLUTNode,
    n=4,
    node_kwargs=node_config,
    layer_config=stable_config
)

# Combined: bit flipping + gradient stabilization
combined_config = LayerConfig(
    flip_probability=0.1,
    grad_stabilization='batchwise',
    grad_target_std=1.0
)
layer = RandomLayer(
    input_size=100,
    output_size=50,
    node_type=LinearLUTNode,
    n=4,
    node_kwargs=node_config,
    layer_config=combined_config
)
```

### Layer Architecture

- Layers use `nn.ModuleList` containing `output_size` independent node instances
- Each node processes 2D tensors: `(batch_size, input_dim) → (batch_size, output_dim)`
- Layer output: `(batch_size, output_size * output_dim_per_node)`
- For typical case with `output_dim=1`: `(batch_size, output_size)`

---

## Nodes

Nodes define the computation at individual LUT units. Each node processes a fixed number of inputs and produces outputs based on learned lookup tables.

### NodeConfig: Type-Safe Configuration

Use `NodeConfig` for type-safe node parameters instead of raw dictionaries:

```python
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY

# Basic configuration
node_config = NodeConfig(
    input_dim=6,        # 6-input LUT (2^6 = 64 table entries)
    output_dim=1,       # Single output per node
)

# With initialization from registry
init_fn = REGISTRY.get_initializer('kaiming_normal')

node_config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=init_fn,
    init_kwargs={'a': 0.0, 'mode': 'fan_in'}
)

# With regularization from registry
reg_fn = REGISTRY.get_regularizer('l2')

node_config = NodeConfig(
    input_dim=6,
    output_dim=1,
    regularizers={'l2': reg_fn}
)

# Node-specific parameters via extra_params
node_config = NodeConfig(
    input_dim=6,
    output_dim=1,
    extra_params={
        'use_cuda': True,
        'alpha': 0.5
    }
)
```

**NodeConfig Parameters:**
- `input_dim`: Number of inputs per node (LUT width)
- `output_dim`: Number of outputs per node (typically 1)
- `regularizers`: Dict mapping regularizer names to functions retrieved from registry (optional)
- `init_fn`: Initialization function retrieved from registry (optional)
- `init_kwargs`: Kwargs for initialization function (optional)
- `extra_params`: Dict for node-specific parameters (optional)

### Available Node Types

| Node Type | Description | Use Case | CUDA Support |
|-----------|-------------|----------|--------------|
| `LinearLUTNode` | Basic LUT with linear table | Simple, interpretable | No |
| `PolyLUTNode` | Polynomial approximation | Smooth functions | No |
| `NeuralLUTNode` | MLP-based LUT | Complex mappings | No |
| `DWNNode` | Differentiable Winner-take-all | Memory efficient | Yes (`efd_cuda`) |
| `DWNStableNode` | Stabilized DWN variant | Production use | Yes (`dwn_stable_cuda`) |
| `ProbabilisticNode` | Probabilistic LUT | Uncertainty modeling | Yes (`probabilistic_cuda`) |
| `FourierNode` | Fourier basis functions | Periodic patterns | Yes (`fourier_cuda`) |
| `HybridNode` | Combined approach | Flexibility | Yes (`hybrid_cuda`) |

### Node Usage Examples

#### LinearLUTNode (Basic)

```python
from difflut.nodes import LinearLUTNode
from difflut.nodes.node_config import NodeConfig

config = NodeConfig(input_dim=4, output_dim=1)
node = LinearLUTNode(**config.to_dict())

# Process batch
x = torch.randint(0, 2, (32, 4)).float()  # Binary inputs
output = node(x)  # (32, 1)
```

#### DWNStableNode (GPU-Accelerated)

```python
from difflut.nodes import DWNStableNode
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY

config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=REGISTRY.get_initializer('kaiming_normal'),
    init_kwargs={'a': 0.0, 'mode': 'fan_in'},
    regularizers={'l2': REGISTRY.get_regularizer('l2')},
    extra_params={'use_cuda': True, 'gradient_scale': 1.0}
)
node = DWNStableNode(**config.to_dict())

x = torch.randn(32, 6).cuda()
output = node(x)  # Uses CUDA kernel if available
```

#### ProbabilisticNode

```python
from difflut.nodes import ProbabilisticNode
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY

config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=REGISTRY.get_initializer('normal'),
    regularizers={'l1': REGISTRY.get_regularizer('l1')},
    extra_params={'temperature': 1.0, 'use_cuda': True}
)
node = ProbabilisticNode(**config.to_dict())

x = torch.randn(32, 6)
output = node(x)  # Probabilistic output
```

#### NeuralLUTNode (MLP-based)

```python
from difflut.nodes import NeuralLUTNode
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY

config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=REGISTRY.get_initializer('kaiming_normal'),
    extra_params={
        'hidden_width': 64,
        'depth': 2,
        'activation': 'relu'
    }
)
node = NeuralLUTNode(**config.to_dict())
```

### Node Initializers

Initializers control how node parameters are initialized. Get them from the registry and pass via `NodeConfig`:

```python
from difflut.registry import REGISTRY
from difflut.nodes.node_config import NodeConfig

# Get initializer from registry
init_fn = REGISTRY.get_initializer('kaiming_normal')

# Create NodeConfig with initializer
config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=init_fn,
    init_kwargs={'a': 0.0, 'mode': 'fan_in', 'nonlinearity': 'relu'}
)

# Create node
NodeClass = REGISTRY.get_node('linear_lut')
node = NodeClass(**config.to_dict())
```

**Available Initializers (from registry):**
- `zeros`: Initialize all weights to 0
- `ones`: Initialize all weights to 1
- `uniform`: Uniform random initialization
- `normal`: Standard normal initialization
- `xavier` or `xavier_uniform`: Xavier/Glorot uniform
- `xavier_normal`: Xavier/Glorot normal
- `kaiming` or `kaiming_uniform`: Kaiming/He uniform
- `kaiming_normal`: Kaiming/He normal
- `variance_stabilized`: Variance stabilizing initialization
- `probabilistic`: Probabilistic initialization

All initializers are accessible via `REGISTRY.get_initializer(name)`.

### Node Regularizers

Regularizers penalize node parameters during training. Get them from the registry and pass via `NodeConfig`:

```python
from difflut.registry import REGISTRY
from difflut.nodes.node_config import NodeConfig

# Get regularizers from registry
l1_reg = REGISTRY.get_regularizer('l1')
l2_reg = REGISTRY.get_regularizer('l2')
spectral_reg = REGISTRY.get_regularizer('spectral')

# Create NodeConfig with multiple regularizers
config = NodeConfig(
    input_dim=6,
    output_dim=1,
    regularizers={
        'l2': l2_reg,
        'spectral': spectral_reg
    }
)

# Create node
NodeClass = REGISTRY.get_node('linear_lut')
node = NodeClass(**config.to_dict())

# Compute regularization loss during training
reg_loss = node.regularization()
total_loss = task_loss + 0.001 * reg_loss
```

**Available Regularizers (from registry):**
- `l1`: L1 weight regularization
- `l2`: L2 weight regularization
- `spectral`: Spectral norm regularization
- `walsh`: Walsh-Hadamard regularization
- `fourier`: Fourier spectrum regularization
- `functional`: Functional regularization

All regularizers are accessible via `REGISTRY.get_regularizer(name)`.
    input_dim=6,
    output_dim=1,
    init_fn=kaiming_normal_init,
    init_kwargs={'a': 0.0, 'mode': 'fan_in', 'nonlinearity': 'relu'}
)

# Variance-stabilized (for Probabilistic nodes)
config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=variance_stabilized_init,
    init_kwargs={'v_target': 1.0}
)
```

**Available Initializers:**
- `zeros_init`, `ones_init`: Constant initialization
- `normal_init`: Normal distribution `N(mean, std)`
- `uniform_init`: Uniform distribution `U(a, b)`
- `xavier_uniform_init`, `xavier_normal_init`: Xavier/Glorot initialization
- `kaiming_uniform_init`, `kaiming_normal_init`: Kaiming/He initialization
- `variance_stabilized_init`: Variance-stabilized for probabilistic nodes

### Node Regularizers

Regularizers encourage desirable properties in learned LUTs. Use them via `NodeConfig`:

```python
from difflut.nodes.utils.regularizers import (
    l_regularizer, l1_regularizer, l2_regularizer,
    spectral_regularizer
)

# L2 functional regularization
config = NodeConfig(
    input_dim=6,
    output_dim=1,
    regularizers={
        'l2': l2_regularizer
    }
)

# Multiple regularizers
config = NodeConfig(
    input_dim=6,
    output_dim=1,
    regularizers={
        'l1': l1_regularizer,
        'spectral': spectral_regularizer
    }
)

# Custom regularization parameters
def custom_l1(node):
    return l1_regularizer(node, num_samples=200)

config = NodeConfig(
    input_dim=6,
    output_dim=1,
    regularizers={'l1': custom_l1}
)
```

**Available Regularizers:**
- `l_regularizer`: General Lp functional regularization (configurable p-norm)
- `l1_regularizer`: L1 functional regularization (sensitivity to bit flips)
- `l2_regularizer`: L2 functional regularization (squared sensitivity)
- `spectral_regularizer`: Spectral/Fourier regularization (encourages low-frequency functions)

**Regularization in Training:**

```python
# Compute regularization loss
model = MyLUTNetwork()
outputs = model(inputs)
classification_loss = criterion(outputs, targets)

# Add regularization from all layers
reg_loss = sum(layer.regularization() for layer in model.layers if hasattr(layer, 'regularization'))

# Total loss
total_loss = classification_loss + 0.01 * reg_loss  # λ = 0.01
total_loss.backward()
```

---

## Modules

Utility modules for post-processing layer outputs.

### GroupSum: Feature Grouping and Summation

`GroupSum` groups layer outputs and sums within groups, typically used as the final layer before classification.

```python
from difflut.utils.modules import GroupSum
import torch

# Create GroupSum
groupsum = GroupSum(
    k=10,                  # Number of output groups (e.g., 10 classes)
    tau=1.0,              # Temperature (divides sum by tau)
    use_randperm=False    # Don't randomly permute features
)

# Input: (batch_size, num_features) from previous layer
x = torch.randn(32, 1000)  # 1000 features from layer

# Output: (batch_size, k) grouped sums
output = groupsum(x)  # (32, 10) for classification
```

**Parameters:**
- `k`: Number of output groups (typically number of classes)
- `tau`: Temperature for scaling output (divides grouped sum)
- `use_randperm`: If True, randomly permute features before grouping

**Usage in Network:**

```python
class LUTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ThermometerEncoder(num_bits=8)
        self.layer1 = RandomLayer(input_size=6272, output_size=1000, ...)
        self.layer2 = RandomLayer(input_size=1000, output_size=1000, ...)
        self.groupsum = GroupSum(k=10, tau=1.0)  # 10 classes
    
    def forward(self, x):
        x = self.encoder(x)
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        x = self.groupsum(x)  # (batch, 10)
        return x
```

**Key Points:**
- Input features should be divisible by `k` (will warn and pad if not)
- Typical use: `k = num_classes` for classification
- Output used directly with `CrossEntropyLoss`

---

## Creating Custom Components

For implementing custom encoders, nodes, layers, or modules, see the [Creating Components Guide](../DEVELOPER_GUIDE/creating_components.md), which covers:
- Extending base classes (`BaseEncoder`, `BaseNode`, `BaseLUTLayer`)
- Registering components with decorators
- Implementing required methods
- Adding CUDA acceleration
- Testing and best practices

---

## Next Steps

- [Registry & Pipelines](registry_pipeline.md) - Learn to build pipelines dynamically
- [Quick Start](../QUICK_START.md) - Run first example
- [Developer Guide](../DEVELOPER_GUIDE.md) - Create custom components
