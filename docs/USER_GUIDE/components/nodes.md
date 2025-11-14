# Nodes Guide

Nodes define computation at individual LUT units. Each node processes a fixed number of binary inputs and produces outputs based on learned lookup tables or learned functions.

---

## Table of Contents
1. [NodeConfig](#nodeconfig)
2. [Initializers](#initializers)
3. [Regularizers](#regularizers)
4. [Available Node Types](#available-node-types)
5. [Node Examples](#node-examples)
6. [GPU Acceleration](#gpu-acceleration)

---

## NodeConfig

Use `NodeConfig` for type-safe node parameters instead of raw dictionaries. It handles structural parameters, initialization, and regularization.

### Basic Configuration

```python
from difflut.nodes.node_config import NodeConfig

# Minimal configuration
config = NodeConfig(
    input_dim=6,        # 6-input LUT (2^6 = 64 table entries)
    output_dim=1,       # Single output per node
)

# With optional parameters
config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=None,                      # Initialization function
    init_kwargs=None,                  # Init function kwargs
    regularizers=None,                 # Regularizer functions
    extra_params=None,                 # Node-specific parameters
)
```

### NodeConfig Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_dim` | int | Number of inputs per node (LUT width, defines 2^n entries) | Required |
| `output_dim` | int | Number of outputs per node | 1 |
| `init_fn` | Callable | Initialization function from registry | None |
| `init_kwargs` | dict | Kwargs passed to initialization function | None |
| `regularizers` | dict | Dict mapping names to regularizer functions | None |
| `extra_params` | dict | Node-specific parameters (e.g., temperature, use_cuda) | None |

### Using NodeConfig with Registry

```python
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY

# Get initializer from registry
init_fn = REGISTRY.get_initializer('kaiming_normal')

# Create config with initializer
config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=init_fn,
    init_kwargs={'a': 0.0, 'mode': 'fan_in', 'nonlinearity': 'relu'}
)

# Get regularizers from registry
l2_reg = REGISTRY.get_regularizer('l2')
spectral_reg = REGISTRY.get_regularizer('spectral')

# Add regularizers
config_with_regs = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=init_fn,
    init_kwargs={'a': 0.0, 'mode': 'fan_in'},
    regularizers={
        'l2': l2_reg,
        'spectral': spectral_reg
    }
)

# Convert to dict for node instantiation
node_kwargs = config_with_regs.to_dict()
NodeClass = REGISTRY.get_node('linear_lut')
node = NodeClass(**node_kwargs)
```

### Node-Specific Parameters

Use `extra_params` for parameters specific to particular node types:

```python
from difflut.nodes.node_config import NodeConfig

# For DWNNode or probabilistic nodes
config = NodeConfig(
    input_dim=6,
    output_dim=1,
    extra_params={
        'use_cuda': True,           # Enable GPU acceleration
        'temperature': 1.0,         # For probabilistic nodes
        'gradient_scale': 1.0,      # For gradient stabilization
        'eval_mode': 'expectation'  # Evaluation mode
    }
)

# For PolyLUT nodes
config = NodeConfig(
    input_dim=6,
    output_dim=1,
    extra_params={
        'degree': 3  # Polynomial degree
    }
)

# For NeuralLUT nodes
config = NodeConfig(
    input_dim=6,
    output_dim=1,
    extra_params={
        'hidden_dim': 64,   # Hidden layer width
        'num_layers': 2     # Number of layers
    }
)
```

---

## Initializers

Initializers control how node parameters (LUT weights) are initialized before training. Always retrieve initializers from the registry:

```python
from difflut.registry import REGISTRY

# List all available initializers
available_inits = REGISTRY.list_initializers()
print("Available initializers:", available_inits)
```

### Available Initializers

| Initializer | Parameters | Best For | Notes |
|-------------|-----------|----------|-------|
| `normal` / `normal_init` | `mean=0.0, std=1.0` | Default initialization | Gaussian distribution |
| `uniform` / `uniform_init` | `a=0.0, b=1.0` | Bounded initialization | Uniform distribution |
| `zeros` / `zeros_init` | None | Zero initialization | All weights = 0 |
| `ones` / `ones_init` | None | Constant initialization | All weights = 1 |
| `xavier_uniform` / `xavier_uniform_init` | `gain=1.0` | Balanced networks | Xavier uniform |
| `xavier_normal` / `xavier_normal_init` | `gain=1.0` | Balanced networks | Xavier normal |
| `kaiming_uniform` / `kaiming_uniform_init` | `a=0, mode='fan_in', nonlinearity='leaky_relu'` | ReLU networks | Kaiming uniform |
| `kaiming_normal` / `kaiming_normal_init` | `a=0, mode='fan_in', nonlinearity='leaky_relu'` | ReLU networks | Kaiming normal |

### Using Initializers

```python
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY

# Get initializer by name
init_fn = REGISTRY.get_initializer('kaiming_normal')

# Create config with initializer
config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=init_fn,
    init_kwargs={
        'a': 0.0,                  # Negative slope
        'mode': 'fan_in',          # fan_in or fan_out
        'nonlinearity': 'relu'     # nonlinearity type
    }
)

# Create node
NodeClass = REGISTRY.get_node('linear_lut')
node = NodeClass(**config.to_dict())

print(f"Node initialized with Kaiming normal")
print(f"Weight shape: {node.weight.shape}")
print(f"Weight stats: mean={node.weight.mean():.4f}, std={node.weight.std():.4f}")
```

### Common Initialization Patterns

```python
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY

# Pattern 1: Small random initialization (conservative)
init_small = REGISTRY.get_initializer('normal')
config_small = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=init_small,
    init_kwargs={'mean': 0.0, 'std': 0.01}  # Very small
)

# Pattern 2: Xavier initialization (balanced)
init_xavier = REGISTRY.get_initializer('xavier_normal')
config_xavier = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=init_xavier,
    init_kwargs={'gain': 1.0}
)

# Pattern 3: Kaiming initialization (for ReLU-like activations)
init_kaiming = REGISTRY.get_initializer('kaiming_normal')
config_kaiming = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=init_kaiming,
    init_kwargs={
        'a': 0.0,
        'mode': 'fan_in',
        'nonlinearity': 'relu'
    }
)

# Pattern 4: Zero initialization (for fine-tuning)
init_zeros = REGISTRY.get_initializer('zeros')
config_zeros = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=init_zeros,
    init_kwargs={}
)
```

---

## Regularizers

Regularizers add constraints during training to improve generalization. Always retrieve regularizers from the registry:

```python
from difflut.registry import REGISTRY

# List all available regularizers
available_regs = REGISTRY.list_regularizers()
print("Available regularizers:", available_regs)
```

### Available Regularizers

| Regularizer | Description | Formula | Use Case |
|-------------|-------------|---------|----------|
| `l1` / `l1_regularizer` | L1 norm penalty | $\lambda \sum \|w\|$ | Sparsity |
| `l2` / `l2_regularizer` | L2 norm penalty | $\lambda \sum w^2$ | Weight decay, smoothness |
| `spectral` / `spectral_regularizer` | Spectral norm penalty | $\lambda \|\sigma_{\max}(W)\|$ | Lipschitz constraint |

### Using Regularizers

```python
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY
import torch

# Get regularizers
l1_reg = REGISTRY.get_regularizer('l1')
l2_reg = REGISTRY.get_regularizer('l2')

# Create config with regularizers
config = NodeConfig(
    input_dim=6,
    output_dim=1,
    regularizers={
        'l1': l1_reg,
        'l2': l2_reg
    }
)

# Create node
NodeClass = REGISTRY.get_node('linear_lut')
node = NodeClass(**config.to_dict())

# Compute regularization loss during training
reg_loss = node.regularization()
print(f"Regularization loss: {reg_loss:.4f}")

# Total loss = task_loss + lambda * reg_loss
task_loss = torch.nn.functional.mse_loss(output, target)
lambda_reg = 0.01
total_loss = task_loss + lambda_reg * reg_loss
```

### Regularization Patterns

```python
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY

# Pattern 1: L2 regularization only (most common)
config_l2 = NodeConfig(
    input_dim=6,
    output_dim=1,
    regularizers={
        'l2': REGISTRY.get_regularizer('l2')
    }
)

# Pattern 2: L1 regularization (sparsity)
config_l1 = NodeConfig(
    input_dim=6,
    output_dim=1,
    regularizers={
        'l1': REGISTRY.get_regularizer('l1')
    }
)

# Pattern 3: Combined L1 + L2 (elastic net)
config_elastic = NodeConfig(
    input_dim=6,
    output_dim=1,
    regularizers={
        'l1': REGISTRY.get_regularizer('l1'),
        'l2': REGISTRY.get_regularizer('l2')
    }
)

# Pattern 4: Spectral normalization (Lipschitz constraint)
config_spectral = NodeConfig(
    input_dim=6,
    output_dim=1,
    regularizers={
        'spectral': REGISTRY.get_regularizer('spectral')
    }
)

# Pattern 5: No regularization
config_none = NodeConfig(
    input_dim=6,
    output_dim=1,
    regularizers=None
)
```

---

## Available Node Types

### LinearLUTNode (Basic, Recommended for beginners)

Simple lookup table with learned weights.

```python
from difflut.nodes import LinearLUTNode
from difflut.nodes.node_config import NodeConfig

config = NodeConfig(input_dim=4, output_dim=1)
node = LinearLUTNode(**config.to_dict())

# Process binary inputs
x = torch.randint(0, 2, (32, 4)).float()
output = node(x)  # (32, 1)
```

**Properties:**
- Fastest inference
- Most memory efficient
- Good baseline model
- No GPU acceleration needed

### PolyLUTNode (Polynomial Approximation)

Uses polynomial basis for smooth approximations.

```python
from difflut.nodes import PolyLUTNode
from difflut.nodes.node_config import NodeConfig

config = NodeConfig(
    input_dim=6,
    output_dim=1,
    extra_params={'degree': 3}  # Polynomial degree
)
node = PolyLUTNode(**config.to_dict())

output = node(x)
```

**Properties:**
- Smoother functions than LinearLUT
- More parameters than LinearLUT
- Better for smooth function approximation
- No GPU acceleration

### NeuralLUTNode (Neural Network based)

Uses small MLP for complex mappings.

```python
from difflut.nodes import NeuralLUTNode
from difflut.nodes.node_config import NodeConfig

config = NodeConfig(
    input_dim=6,
    output_dim=1,
    extra_params={
        'hidden_dim': 64,
        'num_layers': 2
    }
)
node = NeuralLUTNode(**config.to_dict())

output = node(x)
```

**Properties:**
- Most expressive
- Most parameters
- Slowest inference
- Best for complex functions
- No GPU acceleration

### DWNNode (Differentiable Winner Take All)

Memory-efficient GPU-accelerated node.

```python
from difflut.nodes import DWNNode
from difflut.nodes.node_config import NodeConfig

config = NodeConfig(
    input_dim=6,
    output_dim=1,
    extra_params={'use_cuda': True}
)
node = DWNNode(**config.to_dict())

x = torch.randn(32, 6).cuda()
output = node(x)  # Uses CUDA kernel if available
```

**Properties:**
- GPU-accelerated (requires `efd_cuda` extension)
- Memory efficient
- Winner-take-all behavior
- Good for large models

### DWNStableNode (Stabilized DWN)

Improved version of DWN with gradient stabilization.

```python
from difflut.nodes import DWNStableNode
from difflut.nodes.node_config import NodeConfig

config = NodeConfig(
    input_dim=6,
    output_dim=1,
    extra_params={
        'use_cuda': True,
        'gradient_scale': 1.0
    }
)
node = DWNStableNode(**config.to_dict())

output = node(x)
```

**Properties:**
- Improved stability over DWN
- GPU-accelerated
- Better for training deep networks
- Recommended for production

### ProbabilisticNode (Uncertainty Modeling)

Probabilistic LUT for uncertainty quantification.

```python
from difflut.nodes import ProbabilisticNode
from difflut.nodes.node_config import NodeConfig

config = NodeConfig(
    input_dim=6,
    output_dim=1,
    extra_params={
        'temperature': 1.0,
        'use_cuda': True,
        'eval_mode': 'expectation'
    }
)
node = ProbabilisticNode(**config.to_dict())

output = node(x)  # Probabilistic output
```

**Properties:**
- GPU-accelerated
- Provides uncertainty estimates
- Temperature controls randomness
- Good for Bayesian methods

### FourierNode (Fourier Basis)

Uses Fourier basis functions.

```python
from difflut.nodes import FourierNode
from difflut.nodes.node_config import NodeConfig

config = NodeConfig(
    input_dim=6,
    output_dim=1,
    extra_params={'num_freqs': 8}  # Number of frequencies
)
node = FourierNode(**config.to_dict())

output = node(x)
```

**Properties:**
- GPU-accelerated
- Good for periodic patterns
- Efficient spectral representation
- Good for signal processing

### HybridNode (Combined Approach)

Combines multiple node types for flexibility.

```python
from difflut.nodes import HybridNode
from difflut.nodes.node_config import NodeConfig

config = NodeConfig(
    input_dim=6,
    output_dim=1,
    extra_params={
        'use_cuda': True,
        'mix_ratio': 0.5  # Balance between components
    }
)
node = HybridNode(**config.to_dict())

output = node(x)
```

**Properties:**
- GPU-accelerated
- Flexible combination of approaches
- Adaptive behavior
- Best for research

### Comparison

| Node Type | Speed | Memory | GPU | Expressiveness | Use Case |
|-----------|-------|--------|-----|-----------------|----------|
| LinearLUT | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | ⭐⭐ | Baseline, beginner |
| PolyLUT | ⭐⭐⭐ | ⭐⭐⭐ | ❌ | ⭐⭐⭐ | Smooth functions |
| NeuralLUT | ⭐⭐ | ⭐⭐ | ❌ | ⭐⭐⭐⭐⭐ | Complex functions |
| DWNNode | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ⭐⭐⭐ | Large models |
| DWNStable | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ⭐⭐⭐ | Production |
| Probabilistic | ⭐⭐⭐ | ⭐⭐⭐ | ✅ | ⭐⭐⭐ | Uncertainty |
| Fourier | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | ⭐⭐⭐⭐ | Periodic data |
| Hybrid | ⭐⭐⭐ | ⭐⭐ | ✅ | ⭐⭐⭐⭐⭐ | Research |

---

## Node Examples

### Basic Node Creation and Usage

```python
from difflut.nodes import LinearLUTNode, DWNStableNode
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY
import torch

# Example 1: Simple LinearLUT
config = NodeConfig(input_dim=4, output_dim=1)
node = LinearLUTNode(**config.to_dict())

x = torch.randn(32, 4)
y = node(x)
print(f"Output shape: {y.shape}")  # (32, 1)

# Example 2: With initialization
init_fn = REGISTRY.get_initializer('kaiming_normal')
config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=init_fn,
    init_kwargs={'a': 0.0, 'mode': 'fan_in', 'nonlinearity': 'relu'}
)
node = LinearLUTNode(**config.to_dict())

# Example 3: With regularization
l2_reg = REGISTRY.get_regularizer('l2')
config = NodeConfig(
    input_dim=6,
    output_dim=1,
    regularizers={'l2': l2_reg}
)
node = LinearLUTNode(**config.to_dict())

reg_loss = node.regularization()
print(f"Regularization loss: {reg_loss:.4f}")

# Example 4: GPU-accelerated DWNStable
config = NodeConfig(
    input_dim=6,
    output_dim=1,
    extra_params={'use_cuda': True, 'gradient_scale': 1.0}
)
node = DWNStableNode(**config.to_dict())

x_cuda = torch.randn(32, 6).cuda()
y = node(x_cuda)
```

### Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from difflut.nodes import LinearLUTNode
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY

# Setup
config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=REGISTRY.get_initializer('kaiming_normal'),
    init_kwargs={'a': 0.0, 'mode': 'fan_in', 'nonlinearity': 'relu'},
    regularizers={'l2': REGISTRY.get_regularizer('l2')}
)

node = LinearLUTNode(**config.to_dict())
optimizer = optim.Adam(node.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
x = torch.randn(100, 6)
y_target = torch.randn(100, 1)

for epoch in range(10):
    # Forward pass
    y_pred = node(x)
    
    # Loss
    task_loss = criterion(y_pred, y_target)
    reg_loss = node.regularization()
    total_loss = task_loss + 0.01 * reg_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch+1}: Task Loss: {task_loss:.4f}, Reg Loss: {reg_loss:.4f}")
```

---

## GPU Acceleration

Some node types support GPU acceleration through CUDA kernels:

```python
import torch
from difflut.nodes import DWNStableNode
from difflut.nodes.node_config import NodeConfig

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    
    # Create GPU-accelerated node
    config = NodeConfig(
        input_dim=6,
        output_dim=1,
        extra_params={'use_cuda': True}
    )
    node = DWNStableNode(**config.to_dict()).cuda()
    
    # Process on GPU
    x = torch.randn(32, 6, device='cuda')
    y = node(x)
    print(f"Output device: {y.device}")
else:
    print("GPU not available, using CPU")
    config = NodeConfig(input_dim=6, output_dim=1)
    node = LinearLUTNode(**config.to_dict())
```

---

## Next Steps

- **[Layers Guide](layers.md)** - Build layers with nodes
- **[Encoders Guide](encoders.md)** - Encode continuous inputs
- **[Components Overview](components.md)** - High-level reference
