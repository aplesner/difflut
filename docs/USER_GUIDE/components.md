# Components Guide

A comprehensive reference for all DiffLUT components: encoders, nodes, and layers.

---

## Dimension Reference

Quick reference for tensor dimensions through the DiffLUT pipeline:

| Component | Input Shape | Output Shape | Notes |
|-----------|-------------|--------------|-------|
| **Encoder** | `(batch_size, input_dim)` | `(batch_size, input_dim * num_bits)` | Discretizes continuous inputs |
| **Layer** | `(batch_size, input_size)` | `(batch_size, output_size * output_dim)` | Routes inputs to nodes |
| **Layer (internal)** | `(batch_size, input_size)` | `(batch_size, output_size, node_input_dim)` | Maps to node inputs |
| **Nodes** | `(batch_size, output_size, node_input_dim)` | `(batch_size, output_size, node_output_dim)` | Parallel LUT evaluation |
| **GroupSum** | `(batch_size, num_nodes)` | `(batch_size, k)` | Groups & sums features |



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

encoder = ThermometerEncoder(num_bits=8, feature_wise=True)
encoder.fit(train_data)
encoded = encoder(data)
```

**Parameters**:
- `num_bits`: Number of bits per feature
- `feature_wise`: If True, fit each feature independently (recommended)

**Use when**: You want smooth, interpretable discretization.

#### Gray Encoder
Gray code encoding (minimal Hamming distance between consecutive values).

```python
from difflut.encoder import GrayEncoder

encoder = GrayEncoder(num_bits=8, feature_wise=True)
encoder.fit(train_data)
encoded = encoder(data)
```

**Parameters**:
- `num_bits`: Number of bits per feature
- `feature_wise`: If True, fit each feature independently

**Use when**: You want to minimize bit flips during neighboring value transitions.

#### Binary Encoder
Standard binary encoding.

```python
from difflut.encoder import BinaryEncoder

encoder = BinaryEncoder(num_bits=8, feature_wise=True)
encoder.fit(train_data)
encoded = encoder(data)
```

**Parameters**:
- `num_bits`: Number of bits per feature
- `feature_wise`: If True, fit each feature independently

**Use when**: You need standard binary representation.

#### Gaussian Encoder
Gaussian basis functions for smooth encoding.

```python
from difflut.encoder import GaussianEncoder

encoder = GaussianEncoder(num_bits=8, sigma=1.0, feature_wise=True)
encoder.fit(train_data)
encoded = encoder(data)
```

**Parameters**:
- `num_bits`: Number of Gaussian centers
- `sigma`: Standard deviation of Gaussians
- `feature_wise`: If True, fit each feature independently

**Use when**: You want smooth, continuous-like encoding with Gaussian basis functions.

#### One-Hot Encoder
One-hot encoding (sparse representation).

```python
from difflut.encoder import OneHotEncoder

encoder = OneHotEncoder(num_bits=8, feature_wise=True)
encoder.fit(train_data)
encoded = encoder(data)
```

**Use when**: You need sparse, interpretable representations.

#### Logarithmic Encoder
Logarithmic scaling for handling large value ranges.

```python
from difflut.encoder import LogarithmicEncoder

encoder = LogarithmicEncoder(num_bits=8, feature_wise=True)
encoder.fit(train_data)
encoded = encoder(data)
```

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

All nodes accept:
- `input_dim`: List with single integer [n] - number of binary inputs
- `output_dim`: List with single integer [m] - number of outputs
- Other node-specific parameters

### Available Node Types

#### Linear LUT Node
Simple table lookup with differentiable weighting.

```python
from difflut.nodes import LinearLUTNode

node = LinearLUTNode(input_dim=[4], output_dim=[1])

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

node = PolyLUTNode(
    input_dim=[6],
    output_dim=[1],
    degree=3  # Polynomial degree
)
```

**Parameters**:
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

node = NeuralLUTNode(
    input_dim=[4],
    output_dim=[1],
    hidden_width=32,  # MLP hidden layer width
    hidden_layers=2   # Number of hidden layers
)
```

**Parameters**:
- `hidden_width`: Width of hidden MLP layers
- `hidden_layers`: Number of hidden MLP layers

**Characteristics**:
- Most expressive
- Highest computational cost
- Best approximation capability
- Fully differentiable

**Use when**: You need maximum expressiveness.

#### Fourier Node (GPU-accelerated)
Fourier transform-based node with CUDA support.

```python
from difflut.nodes import FourierNode

node = FourierNode(
    input_dim=[4],
    output_dim=[1],
    freq_scale=1.0
)

# Move to GPU for acceleration
node = node.cuda()
```

**Parameters**:
- `freq_scale`: Frequency scaling factor

**Characteristics**:
- GPU-accelerated (CUDA)
- Periodic function learning
- Fast on GPU
- Lower gradients (stabilized)

**Use when**: You have GPU and want fast periodic function learning.

#### Hybrid Node (GPU-accelerated)
Hybrid approach combining LUT and neural components.

```python
from difflut.nodes import HybridNode

node = HybridNode(
    input_dim=[4],
    output_dim=[1],
    hybrid_ratio=0.5
)

# Move to GPU
node = node.cuda()
```

**Parameters**:
- `hybrid_ratio`: Balance between LUT and neural components (0-1)

**Characteristics**:
- GPU-accelerated
- Balanced computation/expressiveness
- Good for large-scale problems
- Faster than pure neural LUTs

**Use when**: You want GPU acceleration with good expressiveness.

#### Gradient Stabilized Node (GPU-accelerated)
Specialized node with gradient normalization for stable training.

```python
from difflut.nodes import GradientStabilizedNode

node = GradientStabilizedNode(
    input_dim=[4],
    output_dim=[1],
    gradient_scale=1.0
)

# Move to GPU
node = node.cuda()
```

**Parameters**:
- `gradient_scale`: Gradient scaling factor for stability

**Characteristics**:
- GPU-accelerated
- Stabilized gradients
- Better convergence
- Especially useful for large networks

**Use when**: You have training stability issues.

#### Probabilistic Node
Probabilistic LUT with uncertainty.

```python
from difflut.nodes import ProbabilisticNode

node = ProbabilisticNode(
    input_dim=[4],
    output_dim=[1],
    uncertainty_scale=0.1
)
```

**Parameters**:
- `uncertainty_scale`: Scale of probabilistic uncertainty

**Characteristics**:
- Provides uncertainty estimates
- Bayesian interpretation
- Useful for Bayesian networks
- Differentiable

**Use when**: You need uncertainty estimates or Bayesian learning.

#### DWN Node
Discrete Wavelet Network node.

```python
from difflut.nodes import DWNNode

node = DWNNode(
    input_dim=[4],
    output_dim=[1]
)
```

**Characteristics**:
- Wavelet-based learning
- Good for multi-scale features
- Efficient representation

**Use when**: You have multi-scale or hierarchical features.

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

#### Grouped Layer
Divides inputs into groups, each group connected to a subset of outputs.

```python
from difflut.layers import GroupedLayer

layer = GroupedLayer(
    input_size=512,
    output_size=256,
    num_groups=8,  # 8 input groups
    node_type=LinearLUTNode,
    n=4,
    node_kwargs={'input_dim': [4], 'output_dim': [1]}
)
```

**Parameters**:
- `num_groups`: Number of input groups

**Characteristics**:
- Semantic/structured connectivity
- Reduces parameter space
- Improves interpretability
- Good for grouped features

**Use when**: Inputs have natural groups (e.g., image regions).

#### Residual Layer
Adds skip connections for deeper networks.

```python
from difflut.layers import ResidualLayer

layer = ResidualLayer(
    input_size=256,
    output_size=256,
    base_layer_type=RandomLayer,
    node_type=LinearLUTNode,
    n=4,
    node_kwargs={'input_dim': [4], 'output_dim': [1]},
    residual_weight=0.5  # Balance between LUT output and skip
)
```

**Parameters**:
- `base_layer_type`: Type of layer to use as base (e.g., RandomLayer)
- `residual_weight`: Weight for combining outputs

**Characteristics**:
- Skip connections
- Easier to train deep networks
- Helps with vanishing gradients
- Input and output size must match

**Use when**: Building deeper networks (3+ LUT layers).

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

### Deep Residual Network

```python
class DeepLUTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ThermometerEncoder(num_bits=8)
        
        # Stack residual layers
        self.res_layer1 = ResidualLayer(128, 128, RandomLayer, LinearLUTNode, 4,
                                        node_kwargs={'input_dim': [4], 'output_dim': [1]})
        self.res_layer2 = ResidualLayer(128, 128, RandomLayer, LinearLUTNode, 4,
                                        node_kwargs={'input_dim': [4], 'output_dim': [1]})
        self.res_layer3 = ResidualLayer(128, 128, RandomLayer, LinearLUTNode, 4,
                                        node_kwargs={'input_dim': [4], 'output_dim': [1]})
        
        self.output = RandomLayer(128, 10, LinearLUTNode, 4,
                                  node_kwargs={'input_dim': [4], 'output_dim': [1]})
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = torch.relu(self.res_layer1(x))
        x = torch.relu(self.res_layer2(x))
        x = torch.relu(self.res_layer3(x))
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
5. **Use residual layers for depth**: Essential for networks with 3+ LUT layers
6. **Experiment with n**: Typically 4-6 inputs per LUT gives good trade-offs

## Next Steps

- [Registry & Pipelines](registry_pipeline.md) - Learn to build pipelines dynamically
- [Quick Start](../QUICK_START.md) - Run first example
- [Developer Guide](../DEVELOPER_GUIDE.md) - Create custom components
