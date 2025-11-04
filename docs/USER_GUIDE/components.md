# Components Guide

A comprehensive reference for all DiffLUT components: encoders, nodes, and layers.

---

## Dimension Reference

Quick reference for tensor dimensions through the DiffLUT pipeline:

| Component | Input Shape | Output Shape (flatten=True) | Output Shape (flatten=False) | Notes |
|-----------|-------------|-----|-----|-------|
| **Encoder** | `(batch_size, input_dim)` | `(batch_size, input_dim * num_bits)` | `(batch_size, input_dim, num_bits)` | Discretizes continuous inputs |
| **Layer** | `(batch_size, input_size)` | `(batch_size, output_size)` | N/A | Routes inputs to `output_size` independent nodes |
| **Single Node** | `(batch_size, node_input_dim)` | `(batch_size, node_output_dim)` | N/A | Each node processes 2D tensors independently |
| **GroupSum** | `(batch_size, num_nodes)` | `(batch_size, k)` | N/A | Groups & sums features |

**Architecture**: Layers use `nn.ModuleList` containing `output_size` independent node instances. Each node processes 2D tensors `(batch, input_dim) → (batch, output_dim)`. The layer iterates through nodes and concatenates outputs.



## Encoders

Encoders transform continuous input values into discrete representations suitable for LUT indexing.

### Why Encoders?

LUT networks require discrete indices to look up values. Encoders discretize continuous inputs:

```
Input: 0.5 (continuous) → Encoder → 1010 (binary) → LUT index
```

### Using Encoders

All encoders follow this pattern:

```python
from difflut.encoder import ThermometerEncoder

# 1. Create encoder
encoder = ThermometerEncoder(num_bits=8)

# 2. Fit to training data (learns min/max values)
train_data = torch.randn(1000, 784)
encoder.fit(train_data)

# 3. Encode inputs
x = torch.randn(32, 784)
encoded = encoder(x)  # Shape: (32, 784 * 8)

# 4. Use in model
output = layer(encoded)
```

### Available Encoders

#### Thermometer Encoder
Unary encoding where each bit represents a threshold.

```python
from difflut.encoder import ThermometerEncoder

# Default: flatten output to 2D
encoder = ThermometerEncoder(num_bits=8)
encoder.fit(train_data)
encoded = encoder(data)  # Shape: (batch_size, input_dim * num_bits)

# Optional: keep output as 3D
encoder = ThermometerEncoder(num_bits=8, flatten=False)
encoder.fit(train_data)
encoded = encoder(data)  # Shape: (batch_size, input_dim, num_bits)
```

**Parameters**:
- `num_bits`: Number of bits per feature
- `flatten`: If True (default), return 2D tensor; if False, return 3D tensor

**Use when**: You want smooth, interpretable discretization.

#### Gray Encoder
Gray code encoding (minimal Hamming distance between consecutive values).


```python
from difflut.encoder import GrayEncoder

# Default: flatten output to 2D
encoder = GrayEncoder(num_bits=8)
encoder.fit(train_data)
encoded = encoder(data)  # Shape: (batch_size, input_dim * num_bits)

# Optional: keep output as 3D
encoder = GrayEncoder(num_bits=8, flatten=False)
encoder.fit(train_data)
encoded = encoder(data)  # Shape: (batch_size, input_dim, num_bits)
```

**Parameters**:
- `num_bits`: Number of bits per feature
- `flatten`: If True (default), return 2D tensor; if False, return 3D tensor

**Use when**: You want to minimize bit flips during neighboring value transitions.

#### Binary Encoder
Standard binary encoding.

```python
from difflut.encoder import BinaryEncoder

# Default: flatten output to 2D
encoder = BinaryEncoder(num_bits=8)
encoder.fit(train_data)
encoded = encoder(data)  # Shape: (batch_size, input_dim * num_bits)

# Optional: keep output as 3D
encoder = BinaryEncoder(num_bits=8, flatten=False)
encoder.fit(train_data)
encoded = encoder(data)  # Shape: (batch_size, input_dim, num_bits)
```

**Parameters**:
- `num_bits`: Number of bits per feature
- `flatten`: If True (default), return 2D tensor; if False, return 3D tensor

**Use when**: You need standard binary representation.

#### Gaussian Thermometer Encoder
Gaussian basis functions for smooth thermometer-like encoding.

```python
from difflut.encoder import GaussianThermometerEncoder

# Default: flatten output to 2D
encoder = GaussianThermometerEncoder(num_bits=8, sigma=1.0)
encoder.fit(train_data)
encoded = encoder(data)  # Shape: (batch_size, input_dim * num_bits)

# Optional: keep output as 3D
encoder = GaussianThermometerEncoder(num_bits=8, sigma=1.0, flatten=False)
encoder.fit(train_data)
encoded = encoder(data)  # Shape: (batch_size, input_dim, num_bits)
```

**Parameters**:
- `num_bits`: Number of Gaussian centers
- `sigma`: Standard deviation of Gaussians
- `flatten`: If True (default), return 2D tensor; if False, return 3D tensor

**Use when**: You want smooth, continuous-like encoding with Gaussian basis functions.

#### Distributive Thermometer Encoder
Distributive thermometer encoding for handling value distributions.

```python
from difflut.encoder import DistributiveThermometerEncoder

# Default: flatten output to 2D
encoder = DistributiveThermometerEncoder(num_bits=8)
encoder.fit(train_data)
encoded = encoder(data)  # Shape: (batch_size, input_dim * num_bits)

# Optional: keep output as 3D
encoder = DistributiveThermometerEncoder(num_bits=8, flatten=False)
encoder.fit(train_data)
encoded = encoder(data)  # Shape: (batch_size, input_dim, num_bits)
```

**Parameters**:
- `num_bits`: Number of bits per feature
- `flatten`: If True (default), return 2D tensor; if False, return 3D tensor

**Use when**: Working with distributed data representations.
#### One-Hot Encoder
One-hot encoding (sparse representation).

```python
from difflut.encoder import OneHotEncoder

# Default: flatten output to 2D
encoder = OneHotEncoder(num_bits=8)
encoder.fit(train_data)
encoded = encoder(data)  # Shape: (batch_size, input_dim * num_bits)

# Optional: keep output as 3D
encoder = OneHotEncoder(num_bits=8, flatten=False)
encoder.fit(train_data)
encoded = encoder(data)  # Shape: (batch_size, input_dim, num_bits)
```

**Parameters**:
- `num_bits`: Number of bins
- `flatten`: If True (default), return 2D tensor; if False, return 3D tensor

**Use when**: You need sparse, interpretable representations.

#### Sign Magnitude Encoder
Sign-magnitude encoding for representing signed values.

```python
from difflut.encoder import SignMagnitudeEncoder

# Default: flatten output to 2D
encoder = SignMagnitudeEncoder(num_bits=8)
encoder.fit(train_data)
encoded = encoder(data)  # Shape: (batch_size, input_dim * num_bits)

# Optional: keep output as 3D
encoder = SignMagnitudeEncoder(num_bits=8, flatten=False)
encoder.fit(train_data)
encoded = encoder(data)  # Shape: (batch_size, input_dim, num_bits)
```

**Parameters**:
- `num_bits`: Number of bits per feature
- `flatten`: If True (default), return 2D tensor; if False, return 3D tensor

**Use when**: Handling signed values or data with clear positive/negative distinction.

#### Logarithmic Encoder
Logarithmic scaling for handling large value ranges.

```python
from difflut.encoder import LogarithmicEncoder

# Default: flatten output to 2D
encoder = LogarithmicEncoder(num_bits=8, base=2.0)
encoder.fit(train_data)
encoded = encoder(data)  # Shape: (batch_size, input_dim * num_bits)

# Optional: keep output as 3D
encoder = LogarithmicEncoder(num_bits=8, base=2.0, flatten=False)
encoder.fit(train_data)
encoded = encoder(data)  # Shape: (batch_size, input_dim, num_bits)
```

**Parameters**:
- `num_bits`: Number of bits per feature
- `base`: Base of logarithm
- `flatten`: If True (default), return 2D tensor; if False, return 3D tensor

**Use when**: Data has exponential or logarithmic characteristics.

### Encoder Fitting Tips

```python
# ✓ Good: Fit on representative sample
sample = torch.cat([train_data[i] for i in range(0, len(train_data), 10)])
encoder.fit(sample)

# ✓ Better: Fit on full training data
encoder.fit(train_data)

# ❌ Wrong: Don't fit on test data
encoder.fit(test_data)  # Data leakage!

# ✓ Reuse fitted encoder
trained_model = load_model()
# Use same encoder that model was trained with
encoded_data = encoder(new_data)
```

---

## Nodes

Nodes define the computation at individual LUT units. Each node processes a fixed number of binary inputs and produces one or more outputs.

### Node Concept

```
Inputs: 4 binary values (0-15 possible combinations)
   ↓
[LUT weights: 16 values]
   ↓
Output: Single value (for 1-output nodes)
```

### Common Node Parameters

All nodes now use type-safe `NodeConfig` for parameter passing:

```python
from difflut.nodes.node_config import NodeConfig

# Create configuration (integers, not lists)
config = NodeConfig(
    input_dim=4,        # Number of binary inputs
    output_dim=1,       # Number of outputs
    init_fn=my_init,    # Optional initialization function
    init_kwargs={},     # Initialization parameters
    extra_params={}     # Node-specific parameters
)
```

**Common parameters:**
- `input_dim`: Integer n - number of binary inputs
- `output_dim`: Integer m - number of outputs
- `init_fn`: Optional initialization function
- `init_kwargs`: Arguments for initialization
- `extra_params`: Dict for node-specific parameters

### Available Node Types

#### Linear LUT Node
Simple table lookup with differentiable weighting.

```python
from difflut.nodes import LinearLUTNode
from difflut.nodes.node_config import NodeConfig

# Create node with 4 inputs and 1 output
config = NodeConfig(input_dim=4, output_dim=1)
node = LinearLUTNode(**config.to_dict())

# 4 inputs → 16 possible combinations
# Learned weights: 16 values
```

**Characteristics**:
- Fastest
- Memory efficient
- Good baseline
- Pure differentiable gradients

**Use when**: You want a simple, fast baseline.

#### Polynomial LUT Node
Polynomial approximation of discrete function.

```python
from difflut.nodes import PolyLUTNode
from difflut.nodes.node_config import NodeConfig

config = NodeConfig(
    input_dim=6,
    output_dim=1,
    extra_params={'degree': 3}
)
node = PolyLUTNode(**config.to_dict())
```

**Parameters** (in extra_params):
- `degree`: Polynomial degree (2-5 recommended)

**Characteristics**:
- Smooth function approximation
- Higher computational cost
- Better generalization
- Differentiable

**Use when**: You want smoother learned functions.

#### Neural LUT Node
Multi-layer perceptron inside LUT cell.

```python
from difflut.nodes import NeuralLUTNode
from difflut.nodes.node_config import NodeConfig

config = NodeConfig(
    input_dim=4,
    output_dim=1,
    extra_params={
        'hidden_width': 32,
        'depth': 2,
        'skip_interval': 2,
        'activation': 'relu',
        'tau_start': 1.0,
        'tau_min': 0.0001,
        'tau_decay_iters': 1000.0,
        'ste': False,
        'grad_factor': 1.0
    }
)
node = NeuralLUTNode(**config.to_dict())
```

**Parameters** (in extra_params):
- `hidden_width`: Width of hidden MLP layers (default: 8)
- `depth`: Number of MLP layers (default: 2)
- `skip_interval`: Interval for skip connections in MLP; 0 = no skips (default: 2)
- `activation`: Activation function - `'relu'`, `'sigmoid'`, or `'leakyrelu'` (default: `'relu'`)
- `tau_start`: Starting temperature for output scaling (default: 1.0)
- `tau_min`: Minimum temperature during decay (default: 0.0001)
- `tau_decay_iters`: Number of iterations for temperature decay (default: 1000.0)
- `ste`: Use Straight-Through Estimator for discretization (default: False)
- `grad_factor`: Gradient scaling factor during backward pass (default: 1.0)

**Characteristics**:
- Most expressive
- Highest computational cost
- Best approximation capability
- Fully differentiable
- Supports temperature annealing for discretization

**Use when**: You need maximum expressiveness or want temperature-based annealing.

#### Fourier Node (GPU-accelerated)
Fourier transform-based node with CUDA support.

```python
from difflut.nodes import FourierNode
from difflut.nodes.node_config import NodeConfig

config = NodeConfig(
    input_dim=4,
    output_dim=1,
    extra_params={
        'use_cuda': True,
        'max_amplitude': 1.0,
        'use_all_frequencies': False
    }
)
node = FourierNode(**config.to_dict())

# Move to GPU for acceleration
node = node.cuda()
```

**Parameters** (in extra_params):
- `use_cuda`: Whether to use CUDA kernels (default: True)
- `max_amplitude`: Maximum amplitude for Fourier coefficients (default: 1.0)
- `use_all_frequencies`: Whether to use all 2^n frequency vectors (default: False)

**Characteristics**:
- GPU-accelerated (CUDA)
- Periodic function learning
- Fast on GPU
- Gradient-stabilized

**Use when**: You have GPU and want fast periodic function learning.

#### Hybrid Node (GPU-accelerated)
Hybrid approach combining LUT and neural components.

```python
from difflut.nodes import HybridNode
from difflut.nodes.node_config import NodeConfig

config = NodeConfig(
    input_dim=4,
    output_dim=1,
    extra_params={
        'use_cuda': True
    }
)
node = HybridNode(**config.to_dict())

# Move to GPU
node = node.cuda()
```

**Parameters** (in extra_params):
- `use_cuda`: Whether to use CUDA kernels for fast GPU acceleration (default: True)

**Characteristics**:
- GPU-accelerated
- Binary forward pass (like DWN) for efficient inference
- Probabilistic backward pass for smooth training
- Good for large-scale problems
- Faster than pure neural LUTs

**Use when**: You want GPU acceleration with good balance between speed and expressiveness.

#### DWN Node
Discrete Wavelet Network node for wavelet-based learning.

```python
from difflut.nodes import DWNNode
from difflut.nodes.node_config import NodeConfig

config = NodeConfig(
    input_dim=4, 
    output_dim=1,
    extra_params={
        'use_cuda': True,
        'clamp_luts': True
    }
)
node = DWNNode(**config.to_dict())
```

**Parameters** (in extra_params):
- `use_cuda`: Whether to use CUDA kernels (default: True)
- `clamp_luts`: Whether to clamp LUT values to [0, 1] (default: True)

**Characteristics**:
- Wavelet-based learning
- Good for multi-scale features
- Efficient representation
- Differentiable gradients

**Use when**: You have multi-scale or hierarchical features.

#### DWN Stable Node (GPU-accelerated)
Gradient-stabilized DWN node with optional CUDA acceleration.

```python
from difflut.nodes import DWNStableNode
from difflut.nodes.node_config import NodeConfig

config = NodeConfig(
    input_dim=4,
    output_dim=1,
    extra_params={
        'use_cuda': True,
        'gradient_scale': 1.25
    }
)
node = DWNStableNode(**config.to_dict())

# Move to GPU for acceleration
node = node.cuda()
```

**Parameters** (in extra_params):
- `use_cuda`: Whether to use CUDA acceleration (default: True)
- `gradient_scale`: Gradient scaling factor for stable training (default: 1.25)

**Characteristics**:
- Gradient-stabilized DWN variant
- Optional GPU acceleration
- Better convergence for large networks
- Stable training with controlled gradient magnitudes

**Use when**: You need stable gradient flow or have GPU available.

#### Probabilistic Node
Probabilistic LUT with uncertainty estimation.

```python
from difflut.nodes import ProbabilisticNode
from difflut.nodes.node_config import NodeConfig

config = NodeConfig(
    input_dim=4,
    output_dim=1,
    extra_params={
        'temperature': 1.0,
        'eval_mode': 'expectation',
        'use_cuda': True
    }
)
node = ProbabilisticNode(**config.to_dict())
```

**Parameters** (in extra_params):
- `temperature`: Temperature for sigmoid scaling (default: 1.0)
- `eval_mode`: Evaluation mode - `'expectation'`, `'deterministic'`, or `'threshold'` (default: `'expectation'`)
  - `'expectation'`: Expected value computation
  - `'deterministic'`: Binary thresholding at 0.5
  - `'threshold'`: Threshold-based evaluation
- `use_cuda`: Whether to use CUDA kernels for acceleration (default: True)

**Characteristics**:
- Provides probabilistic outputs and uncertainty estimates
- Supports multiple evaluation modes
- GPU acceleration available
- Bayesian interpretation for uncertainty quantification
- Differentiable

**Use when**: You need uncertainty estimates or probabilistic outputs.

---

## Layers

Layers define how inputs connect to LUT nodes, creating network layers with different connectivity patterns.

### Layer Concept

```
Inputs (100 features)
    ↓
[Layer: Route 100 inputs to 32 LUT nodes]
    ↓
Each node gets specific input subset
    ↓
Output (32 values)
```

### Common Layer Parameters

All layers accept:
- `input_size`: Number of input features
- `output_size`: Number of output features (LUT nodes)
- `node_type`: Node class to use (not string)
- `n`: Number of inputs per LUT
- `node_kwargs`: Dict of parameters to pass to nodes
- `flip_probability` (optional): Probability of bit flips during training for augmentation (default: 0.0)
- `grad_stabilization` (optional): Gradient stabilization mode - `'none'`, `'layerwise'`, or `'batchwise'` (default: `'none'`)
  - `'none'`: No gradient stabilization
  - `'layerwise'`: Normalize gradients per layer
  - `'batchwise'`: Normalize gradients per batch
- `grad_target_std` (optional): Target standard deviation for gradient stabilization (default: 1.0)
- `grad_subtract_mean` (optional): Whether to subtract mean from gradients (default: False)
- `grad_epsilon` (optional): Small constant for numerical stability in gradient normalization (default: 1e-8)

**Example with layer parameters:**
```python
from difflut.layers import RandomLayer
from difflut.nodes import LinearLUTNode
from difflut.nodes.node_config import NodeConfig

# Configure with training augmentation and gradient stabilization
layer = RandomLayer(
    input_size=512,
    output_size=256,
    node_type=LinearLUTNode,
    n=4,
    flip_probability=0.01,  # 1% bit flip augmentation
    grad_stabilization='layerwise',
    grad_target_std=1.0,
    node_kwargs=NodeConfig(input_dim=4, output_dim=1)
)
```

### Available Layer Types

#### Random Layer
Random connectivity - each LUT node receives n random inputs.

```python
from difflut.layers import RandomLayer
from difflut.nodes import LinearLUTNode

layer = RandomLayer(
    input_size=512,
    output_size=256,
    node_type=LinearLUTNode,
    n=4,
    node_kwargs={'input_dim': [4], 'output_dim': [1]}
)
```

**Characteristics**:
- Simple and fast
- No learnable routing
- Good baseline
- Memory efficient

**Use when**: You want a simple baseline layer.

#### Learnable Layer
Learns which inputs each LUT node receives.

```python
from difflut.layers import LearnableLayer

layer = LearnableLayer(
    input_size=512,
    output_size=256,
    node_type=LinearLUTNode,
    n=4,
    node_kwargs={'input_dim': [4], 'output_dim': [1]}
)

# LearnableLayer has learnable routing parameters
# These are optimized during training
```

**Characteristics**:
- Learns connectivity pattern
- More expressive
- More parameters
- Slower to train

**Use when**: You want to learn input connectivity.

---

## Stacking Components into Networks

### Simple Sequential Network

```python
import torch.nn as nn
from difflut.encoder import ThermometerEncoder
from difflut.layers import RandomLayer
from difflut.nodes import LinearLUTNode

class SimpleLUT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ThermometerEncoder(num_bits=8)
        self.layer1 = RandomLayer(784*8, 128, LinearLUTNode, 4,
                                  node_kwargs={'input_dim': [4], 'output_dim': [1]})
        self.layer2 = RandomLayer(128, 10, LinearLUTNode, 4,
                                  node_kwargs={'input_dim': [4], 'output_dim': [1]})
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x
```

### Deep Network with Multiple Layers

```python
class DeepLUTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ThermometerEncoder(num_bits=8)
        
        # Stack multiple learnable layers for better representation learning
        self.layer1 = LearnableLayer(784*8, 256, LinearLUTNode, 4,
                                        node_kwargs={'input_dim': [4], 'output_dim': [1]})
        self.layer2 = LearnableLayer(256, 128, LinearLUTNode, 4,
                                        node_kwargs={'input_dim': [4], 'output_dim': [1]})
        self.layer3 = LearnableLayer(128, 64, LinearLUTNode, 4,
                                        node_kwargs={'input_dim': [4], 'output_dim': [1]})
        
        self.output = RandomLayer(64, 10, LinearLUTNode, 4,
                                  node_kwargs={'input_dim': [4], 'output_dim': [1]})
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.output(x)
        return x
```

### Mixed Node Types

```python
class MixedNodeLUT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ThermometerEncoder(num_bits=8)
        
        # Linear LUTs for fast computation
        self.layer1 = RandomLayer(784*8, 256, LinearLUTNode, 4,
                                  node_kwargs={'input_dim': [4], 'output_dim': [1]})
        
        # Polynomial LUTs for better fitting
        self.layer2 = RandomLayer(256, 128, PolyLUTNode, 6,
                                  node_kwargs={'input_dim': [6], 'output_dim': [1], 'degree': 3})
        
        # Neural LUTs for output layer
        self.layer3 = RandomLayer(128, 10, NeuralLUTNode, 4,
                                  node_kwargs={'input_dim': [4], 'output_dim': [1], 'hidden_width': 32})
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
```

## Best Practices

1. **Start simple**: Use LinearLUTNode + RandomLayer as baseline
2. **Fit encoders properly**: Always fit on representative training data
3. **Match layer dimensions**: Output size of layer n = input size of layer n+1
4. **Use GPU nodes for large networks**: Fourier, Hybrid for speed
5. **Use LearnableLayer for depth**: Learns better connectivity than Random for deeper networks
6. **Experiment with n**: Typically 4-6 inputs per LUT gives good trade-offs

## Next Steps

- [Registry & Pipelines](registry_pipeline.md) - Learn to build pipelines dynamically
- [Quick Start](../QUICK_START.md) - Run first example
- [Developer Guide](../DEVELOPER_GUIDE.md) - Create custom components
