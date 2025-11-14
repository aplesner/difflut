# Layers Guide

Layers define connectivity patterns between encoded inputs and LUT nodes. DiffLUT provides two main layer types: `RandomLayer` (fixed random connectivity) and `LearnableLayer` (trainable connectivity).

---

## Table of Contents
1. [LayerConfig](#layerconfig)
2. [RandomLayer](#randomlayer)
3. [LearnableLayer](#learnablelayer)
4. [Layer Architecture](#layer-architecture)
5. [Training Configurations](#training-configurations)
6. [Examples](#examples)

---

## LayerConfig

Use `LayerConfig` for type-safe layer training parameters. These parameters affect how the layer behaves during training (e.g., bit flipping, gradient stabilization).

### Basic Configuration

```python
from difflut.layers.layer_config import LayerConfig

# Minimal configuration (no augmentation)
config = LayerConfig()

# Full configuration
config = LayerConfig(
    flip_probability=0.1,           # Bit flip augmentation
    grad_stabilization='layerwise',  # Gradient normalization
    grad_target_std=1.0,             # Target gradient std
    grad_subtract_mean=False,        # Center gradients before scaling
    grad_epsilon=1e-8                # Numerical stability
)
```

### LayerConfig Parameters

| Parameter | Type | Default | Description | Range |
|-----------|------|---------|-------------|-------|
| `flip_probability` | float | 0.0 | Probability of flipping each bit during training (augmentation) | [0.0, 1.0] |
| `grad_stabilization` | str | `'none'` | Gradient normalization mode | `'none'`, `'layerwise'`, `'batchwise'` |
| `grad_target_std` | float | 1.0 | Target standard deviation for gradient rescaling | > 0 |
| `grad_subtract_mean` | bool | False | Whether to center gradients before rescaling | True/False |
| `grad_epsilon` | float | 1e-8 | Small constant for numerical stability | > 0 |

### Parameter Meanings

**`flip_probability`**: Augmentation that randomly flips bits during forward pass
```python
# No augmentation (original inputs)
config = LayerConfig(flip_probability=0.0)

# 10% bit flip augmentation (robust training)
config = LayerConfig(flip_probability=0.1)

# 50% bit flip augmentation (extreme augmentation)
config = LayerConfig(flip_probability=0.5)
```

**`grad_stabilization`**: How to normalize gradients across the layer
```python
# No normalization (default)
config = LayerConfig(grad_stabilization='none')

# Layerwise normalization (across entire layer)
config = LayerConfig(grad_stabilization='layerwise')

# Batchwise normalization (per batch element)
config = LayerConfig(grad_stabilization='batchwise')
```

---

## RandomLayer

Creates fixed random connections between inputs and nodes. Each connection is established randomly and remains fixed (no learning of connections).

### Basic Usage

```python
from difflut.layers import RandomLayer
from difflut.nodes import LinearLUTNode
from difflut.nodes.node_config import NodeConfig
from difflut.layers.layer_config import LayerConfig

# Create node configuration
node_config = NodeConfig(
    input_dim=4,       # 4 inputs per node (2^4 = 16 LUT entries)
    output_dim=1       # 1 output per node
)

# Create layer configuration (optional - uses defaults if not provided)
layer_config = LayerConfig(flip_probability=0.0)

# Create layer
layer = RandomLayer(
    input_size=100,        # Number of input features
    output_size=50,        # Number of nodes to create
    node_type=LinearLUTNode,
    n=4,                   # Number of inputs per node (must match node_config.input_dim)
    node_kwargs=node_config,
    layer_config=layer_config,
    seed=42                # Random seed for reproducibility
)

# Forward pass
x = torch.randn(32, 100)   # Batch of 32 samples, 100 features
output = layer(x)
print(f"Output shape: {output.shape}")  # (32, 50) = 32 batch_size * 50 output_size
```

### RandomLayer Parameters

| Parameter | Type | Description | Notes |
|-----------|------|-------------|-------|
| `input_size` | int | Number of input features from previous layer or encoder | e.g., encoded_dim |
| `output_size` | int | Number of LUT nodes to create | Output = batch_size × output_size × output_dim |
| `node_type` | Type | Node class (e.g., `LinearLUTNode`, `DWNStableNode`) | From registry via `REGISTRY.get_node()` |
| `n` | int | Number of inputs per node | Must match `node_config.input_dim` |
| `node_kwargs` | NodeConfig | Node configuration object | See [Nodes Guide](nodes.md) |
| `layer_config` | LayerConfig | Layer training parameters | Optional, uses defaults if None |
| `seed` | int | Random seed for reproducible connectivity | Default: 42 |

### Connectivity Pattern

RandomLayer creates a fixed random mapping:
- Each node receives `n` random inputs from the `input_size` features
- Inputs are selected without replacement (each input used at least once before reuse)
- Mapping is determined by the seed and remains fixed after layer creation

```python
# Example: Understanding connectivity

layer = RandomLayer(
    input_size=10,      # 10 input features
    output_size=3,      # 3 nodes
    node_type=LinearLUTNode,
    n=4,                # Each node gets 4 inputs
    node_kwargs=NodeConfig(input_dim=4, output_dim=1),
    seed=42
)

# Internal structure:
# - Node 0: Receives 4 random inputs from the 10 inputs
# - Node 1: Receives 4 random inputs from the 10 inputs
# - Node 2: Receives 4 random inputs from the 10 inputs
# - These selections are determined by the seed

# Forward pass routing is fixed:
x = torch.randn(32, 10)  # 32 samples, 10 features
output = layer(x)         # (32, 3) - routes through fixed connections
```

---

## LearnableLayer

Learns optimal connections between inputs and nodes during training using Gumbel-Softmax sampling.

### Basic Usage

```python
from difflut.layers import LearnableLayer
from difflut.nodes import LinearLUTNode
from difflut.nodes.node_config import NodeConfig
from difflut.layers.layer_config import LayerConfig

# Create node configuration
node_config = NodeConfig(input_dim=4, output_dim=1)

# Create layer
layer = LearnableLayer(
    input_size=100,
    output_size=50,
    node_type=LinearLUTNode,
    n=4,
    node_kwargs=node_config,
    layer_config=LayerConfig(),
    temperature=1.0,  # Gumbel-Softmax temperature
    hard=False        # Use soft sampling during training
)

# Forward pass
x = torch.randn(32, 100)
output = layer(x)
print(f"Output shape: {output.shape}")  # (32, 50)
```

### LearnableLayer Parameters

All parameters from `RandomLayer`, plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Gumbel-Softmax temperature (higher = softer sampling) |
| `hard` | bool | False | If True: use hard one-hot; if False: use soft differentiable sampling |

### Temperature Effect

```python
# Temperature controls sampling hardness in Gumbel-Softmax

# Low temperature (0.1) - nearly hard one-hot (sharp decisions)
layer_hard = LearnableLayer(..., temperature=0.1, hard=False)

# Medium temperature (1.0) - balanced (default)
layer_balanced = LearnableLayer(..., temperature=1.0, hard=False)

# High temperature (10.0) - very soft (smooth gradients)
layer_soft = LearnableLayer(..., temperature=10.0, hard=False)

# Hard one-hot (no gradient through discrete sampling)
layer_hard_onehot = LearnableLayer(..., hard=True)
```

---

## Layer Architecture

Understanding the internal structure helps with debugging and optimization:

### Tensor Flow Through Layer

```python
import torch
from difflut.layers import RandomLayer
from difflut.nodes.node_config import NodeConfig

# Create layer
node_config = NodeConfig(input_dim=6, output_dim=1)
layer = RandomLayer(
    input_size=300,    # 300 input features (e.g., 100 features × 3 bits encoder)
    output_size=128,   # 128 nodes
    node_type=LinearLUTNode,
    n=6,               # 6 inputs per node
    node_kwargs=node_config
)

# Forward pass tracking
batch_size = 32
x = torch.randn(batch_size, 300)

# Internal steps:
# 1. RandomLayer receives: (batch_size=32, input_size=300)
# 2. For each of 128 nodes:
#    - Select n=6 inputs from the 300 features (fixed random selection)
#    - Pass to LinearLUTNode: (batch_size=32, n=6) → (batch_size=32, output_dim=1)
# 3. Concatenate all node outputs: (batch_size=32, output_size=128)

output = layer(x)
print(f"Output shape: {output.shape}")  # (32, 128)

# Module structure
print(f"Number of nodes: {len(layer.nodes)}")  # 128
print(f"First node type: {type(layer.nodes[0])}")  # LinearLUTNode
print(f"First node weight shape: {layer.nodes[0].weight.shape}")  # Depends on node_input_dim
```

### Key Points

- **Independent Nodes**: Each node is an independent `nn.Module` in an `nn.ModuleList`
- **Fixed Connectivity**: Connections established at layer creation (seed-determined)
- **Batch Processing**: All nodes process in parallel
- **Output Concatenation**: Individual node outputs concatenated to form layer output

---

## Training Configurations

Different training scenarios require different layer configurations:

### Configuration 1: No Augmentation (Baseline)

```python
from difflut.layers.layer_config import LayerConfig

# Default configuration - no training augmentation
config = LayerConfig()

# Equivalent to:
config = LayerConfig(
    flip_probability=0.0,
    grad_stabilization='none'
)

# Use when:
# - Baseline training
# - Clean data
# - Training without robustness concerns
```

### Configuration 2: Robust Training with Bit Flipping

```python
# Add bit flip augmentation for robustness
config = LayerConfig(flip_probability=0.1)  # 10% bit flips

# Use when:
# - Want robust models
# - Concerned about noisy inputs
# - Doing adversarial training
```

### Configuration 3: Gradient Stabilization

```python
# Stabilize gradients for deep networks
config = LayerConfig(
    grad_stabilization='layerwise',
    grad_target_std=1.0
)

# Use when:
# - Training deep networks (many layers)
# - Experiencing gradient explosion/vanishing
# - Using many layers in sequence
```

### Configuration 4: Combined Robust + Stable

```python
# Both augmentation and gradient stabilization
config = LayerConfig(
    flip_probability=0.1,
    grad_stabilization='layerwise',
    grad_target_std=1.0,
    grad_subtract_mean=False,
    grad_epsilon=1e-8
)

# Use when:
# - Want both robustness and stability
# - Training large deep networks with noisy data
# - Production models
```

### Configuration 5: Aggressive Augmentation

```python
# Heavy bit flipping for extreme robustness
config = LayerConfig(flip_probability=0.5)

# Use when:
# - Very noisy data
# - Want maximum robustness
# - Testing robustness limits
```

---

## Examples

### Example 1: Simple Two-Layer Network

```python
import torch
import torch.nn as nn
from difflut.layers import RandomLayer
from difflut.nodes import LinearLUTNode
from difflut.nodes.node_config import NodeConfig
from difflut.layers.layer_config import LayerConfig

class SimpleLUTNet(nn.Module):
    def __init__(self, input_size=100, hidden_size=64, output_size=10):
        super().__init__()
        
        node_config = NodeConfig(input_dim=4, output_dim=1)
        
        # Layer 1
        self.layer1 = RandomLayer(
            input_size=input_size,
            output_size=hidden_size,
            node_type=LinearLUTNode,
            n=4,
            node_kwargs=node_config,
            seed=42
        )
        
        # Layer 2
        self.layer2 = RandomLayer(
            input_size=hidden_size,
            output_size=output_size,
            node_type=LinearLUTNode,
            n=4,
            node_kwargs=node_config,
            seed=43
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Usage
model = SimpleLUTNet(input_size=100, hidden_size=64, output_size=10)
x = torch.randn(32, 100)
output = model(x)
print(f"Output shape: {output.shape}")  # (32, 10)
```

### Example 2: Deep Network with Gradient Stabilization

```python
import torch
import torch.nn as nn
from difflut.layers import RandomLayer
from difflut.nodes import DWNStableNode
from difflut.nodes.node_config import NodeConfig
from difflut.layers.layer_config import LayerConfig

class DeepLUTNet(nn.Module):
    def __init__(self, depth=5, layer_size=128):
        super().__init__()
        
        # Gradient stabilization for deep networks
        layer_config = LayerConfig(
            grad_stabilization='layerwise',
            grad_target_std=1.0
        )
        
        node_config = NodeConfig(
            input_dim=6,
            output_dim=1,
            extra_params={'use_cuda': torch.cuda.is_available()}
        )
        
        # Create deep stack
        self.layers = nn.ModuleList()
        input_size = layer_size  # Assume encoder output
        
        for i in range(depth):
            layer = RandomLayer(
                input_size=input_size,
                output_size=layer_size,
                node_type=DWNStableNode,
                n=6,
                node_kwargs=node_config,
                layer_config=layer_config,
                seed=42 + i
            )
            self.layers.append(layer)
            input_size = layer_size
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Usage
model = DeepLUTNet(depth=5, layer_size=128)
x = torch.randn(32, 128)
output = model(x)
```

### Example 3: Learnable Connectivity

```python
import torch
import torch.nn as nn
from difflut.layers import LearnableLayer
from difflut.nodes import LinearLUTNode
from difflut.nodes.node_config import NodeConfig

class AdaptiveLUTNet(nn.Module):
    def __init__(self, input_size=100, hidden_size=64):
        super().__init__()
        
        node_config = NodeConfig(input_dim=4, output_dim=1)
        
        # Learnable connectivity layer
        self.layer = LearnableLayer(
            input_size=input_size,
            output_size=hidden_size,
            node_type=LinearLUTNode,
            n=4,
            node_kwargs=node_config,
            temperature=1.0,  # Moderate temperature
            hard=False        # Soft sampling during training
        )
    
    def forward(self, x):
        return self.layer(x)

# Usage
model = AdaptiveLUTNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

x = torch.randn(32, 100)
target = torch.randn(32, 64)

for epoch in range(10):
    output = model(x)
    loss = nn.MSELoss()(output, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

### Example 4: Custom Augmentation During Training

```python
import torch
import torch.nn as nn
from difflut.layers import RandomLayer
from difflut.nodes import LinearLUTNode
from difflut.nodes.node_config import NodeConfig
from difflut.layers.layer_config import LayerConfig
from difflut.registry import REGISTRY

class RobustLUTNet(nn.Module):
    def __init__(self, input_size=100, hidden_size=64, training_robust=True):
        super().__init__()
        
        # Different configs for training vs inference
        if training_robust:
            layer_config = LayerConfig(flip_probability=0.1)
        else:
            layer_config = LayerConfig(flip_probability=0.0)
        
        node_config = NodeConfig(
            input_dim=4,
            output_dim=1,
            init_fn=REGISTRY.get_initializer('kaiming_normal'),
            init_kwargs={'a': 0.0, 'mode': 'fan_in'},
            regularizers={'l2': REGISTRY.get_regularizer('l2')}
        )
        
        self.layer = RandomLayer(
            input_size=input_size,
            output_size=hidden_size,
            node_type=LinearLUTNode,
            n=4,
            node_kwargs=node_config,
            layer_config=layer_config
        )
    
    def forward(self, x):
        return self.layer(x)

# Usage
model = RobustLUTNet(training_robust=True)
model.train()  # Enable dropout/augmentation
x_train = torch.randn(32, 100)
output_train = model(x_train)

model.eval()   # Disable dropout/augmentation
x_test = torch.randn(32, 100)
output_test = model(x_test)
```

---

## Next Steps

- **[Nodes Guide](nodes.md)** - Understand node types and configuration
- **[Encoders Guide](encoders.md)** - Encode continuous inputs
- **[Components Overview](components.md)** - High-level reference
