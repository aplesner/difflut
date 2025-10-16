
# DiffLUT Library

DiffLUT is a modular PyTorch library for differentiable Look-Up Table (LUT) neural networks, designed for efficient FPGA deployment and research. It provides encoders, LUT nodes, flexible layers, and CUDA acceleration.

## Overview


**Features:**
- Modular LUT nodes and layers
- Multiple input encoders
- CUDA acceleration (optional)
- FPGA export tools


## Installation

```bash
pip install -e .
```
Requires Python 3.7+, PyTorch 1.9+, and (optionally) CUDA for GPU support.


CUDA kernels (Fourier, Hybrid, GradientStabilized nodes) are compiled automatically if CUDA is available. Otherwise, CPU fallback is used.



## Usage Overview


### Encoders

Encoders transform continuous input values into discrete representations suitable for LUT indexing.

```python
from difflut.encoder import ThermometerEncoder
import torch

# Create a 4-bit thermometer encoder
encoder = ThermometerEncoder(num_bits=4)

# Fit encoder to data (computes min/max values)
train_data = torch.randn(1000, 784)  # Training data
encoder.fit(train_data)

# Now encode inputs
x = torch.randn(32, 784)  # Batch of MNIST images
encoded = encoder(x)       # Shape: (32, 784 * 4)
```


Encoders transform continuous inputs for LUT indexing. Types: Thermometer, Gaussian, Gray, OneHot, Binary, Logarithmic, etc.


### Nodes

Nodes define the behavior of individual LUTs, including:
- Forward pass computation (training vs. inference)
- Gradient calculation strategies
- Weight parametrization
- Hardware export formats

```python
from difflut.nodes import LinearLUTNode

# Create a LUT node with 4 inputs
node = LinearLUTNode(input_dim=[4], output_dim=[1])

# Use in forward pass
inputs = torch.randint(0, 2, (32, 4))  # Binary inputs
output = node(inputs)  # Shape: (32,)
```


LUT nodes define computation and gradients. Types: LinearLUTNode, PolyLUTNode, NeuralLUTNode, DWNNode, ProbabilisticNode, FourierNode, HybridNode, GradientStabilizedNode.


### Layers

Layers define how input features connect to LUT nodes, creating complete network layers.

```python
from difflut.layers import RandomLayer
from difflut.nodes import LinearLUTNode

# Create a layer with random connectivity
layer = RandomLayer(
    input_size=784,           # MNIST flattened
    output_size=128,          # Hidden layer size
    node_type=LinearLUTNode,  # Pass the class, not string
    n=4,                      # Number of inputs per LUT
    node_kwargs={'input_dim': [4], 'output_dim': [1]}
)

# Use like any PyTorch module
x = torch.randn(32, 784)
output = layer(x)  # Shape: (32, 128)
```


Layers connect features to LUT nodes. Types: RandomLayer, LearnableLayer, GroupedLayer, ResidualLayer.




## Quick Start

```python
import torch
import torch.nn as nn
from difflut.encoder import ThermometerEncoder
from difflut.layers import RandomLayer

class SimpleLUTNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        
        # Encoder: 8 bits per input feature
        self.encoder = ThermometerEncoder(num_bits=8)
        encoded_size = input_size * 8
        
        # Hidden layer: Random connectivity with 4-input LUTs
        self.hidden = RandomLayer(
            input_size=encoded_size,
            output_size=hidden_size,
            node_type=LinearLUTNode,
            n=4,
            node_kwargs={'input_dim': [4], 'output_dim': [1]}
        )
        
        # Output layer
        self.output = RandomLayer(
            input_size=hidden_size,
            output_size=num_classes,
            node_type=LinearLUTNode,
            n=4,
            node_kwargs={'input_dim': [4], 'output_dim': [1]}
        )
    
    def forward(self, x):
        # Flatten and encode
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        
        # LUT layers
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# Create and use the model
model = SimpleLUTNetwork()
x = torch.randn(32, 1, 28, 28)  # Batch of MNIST images
output = model(x)
print(f"Output shape: {output.shape}")  # (32, 10)
```


## Advanced Usage
See the full documentation for details on encoder fitting, custom node/layer creation, and hardware export.

### Encoder Fitting and Usage

Encoders need to be fitted to your data to learn the appropriate value ranges:

```python
from difflut.encoder import ThermometerEncoder, GrayEncoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                               transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

# Get a sample of data for fitting
sample_data = next(iter(train_loader))[0]
sample_data = sample_data.view(sample_data.size(0), -1)  # Flatten

# Create and fit encoder
encoder = ThermometerEncoder(num_bits=8)
encoder.fit(sample_data)

# Now the encoder is ready to use
# You can save the fitted encoder and reuse it
encoded_sample = encoder(sample_data)
print(f"Original shape: {sample_data.shape}")    # (1000, 784)
print(f"Encoded shape: {encoded_sample.shape}")  # (1000, 784*8)

# For different encoder types
gray_encoder = GrayEncoder(num_bits=8, feature_wise=True)
gray_encoder.fit(sample_data)
gray_encoded = gray_encoder(sample_data)
```

### Custom Node Implementation

Create your own LUT node by extending `BaseNode`:

```python
from difflut.nodes import BaseNode
from difflut import register_node

@register_node('MyCustomNode')
class CustomLUTNode(BaseNode):
    def __init__(self, input_dim=None, output_dim=None, **kwargs):
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)
        # Initialize your custom parameters
        # num_inputs is computed from input_dim
        self.custom_weights = nn.Parameter(torch.randn(2**self.num_inputs, self.num_outputs))
    
    def forward_train(self, x):
        """Forward pass during training"""
        # Implement differentiable forward pass
        indices = self._binary_to_index(x)
        return self.custom_weights[indices]
    
    def forward_eval(self, x):
        """Forward pass during evaluation (can be quantized)"""
        return self.forward_train(x)
    
    def backward_input(self, grad_output):
        """Custom gradient w.r.t inputs"""
        # Return input gradients
        return grad_output
    
    def backward_weights(self, grad_output):
        """Custom gradient w.r.t weights"""
        # Update self.custom_weights.grad
        pass
    
    def regularization(self):
        """Compute regularization loss"""
        return 0.0
    
    def export_bitstream(self):
        """Export LUT configuration for FPGA"""
        return self.custom_weights.detach().cpu().numpy()
```

### Using Different Layer Types

```python
from difflut.layers import LearnableLayer, GroupedLayer, ResidualLayer, RandomLayer
from difflut.nodes import PolyLUTNode, NeuralLUTNode, GradientStabilizedNode

# Learnable connectivity - learns which inputs to use
learnable_layer = LearnableLayer(
    input_size=512,
    output_size=256,
    node_type=PolyLUTNode,
    n=6,
    node_kwargs={'input_dim': [6], 'output_dim': [1], 'degree': 3}
)

# Grouped connectivity - semantic input grouping
grouped_layer = GroupedLayer(
    input_size=512,
    output_size=256,
    num_groups=8,
    node_type=NeuralLUTNode,
    n=4,
    node_kwargs={'input_dim': [4], 'output_dim': [1], 'hidden_width': 32}
)

# Residual layer - adds skip connections for deeper networks
residual_layer = ResidualLayer(
    input_size=256,
    output_size=256,
    base_layer_type=RandomLayer,
    node_type=GradientStabilizedNode,
    n=4,
    node_kwargs={'input_dim': [4], 'output_dim': [1], 'gradient_scale': 1.0},
    residual_weight=0.5
)
```


## API Reference
See code docstrings and examples for detailed API usage.

### Encoder Classes

All encoders require calling `.fit(data)` before encoding, or providing min/max values directly.

#### `ThermometerEncoder(num_bits=8, threshold_values=None)`
- `num_bits`: Number of bits for encoding
- `threshold_values`: Optional pre-defined thresholds

#### `GaussianThermometerEncoder(num_bits=8, sigma=1.0)`
- `num_bits`: Number of bits for encoding
- `sigma`: Gaussian spread parameter

#### `DistributiveThermometerEncoder(num_bits=8, distribution='uniform')`
- `num_bits`: Number of bits for encoding
- `distribution`: Distribution type for threshold placement

#### `GrayEncoder(num_bits=8, feature_wise=True)`
- `num_bits`: Number of bits for encoding
- `feature_wise`: If True, compute ranges per feature

#### `OneHotEncoder(num_classes=10)`
- `num_classes`: Number of classes/categories

#### `BinaryEncoder(num_bits=8, feature_wise=True)`
- `num_bits`: Number of bits for encoding
- `feature_wise`: If True, compute ranges per feature

#### `SignMagnitudeEncoder(num_bits=8, feature_wise=True)`
- `num_bits`: Number of bits for encoding (includes sign bit)
- `feature_wise`: If True, compute ranges per feature

#### `LogarithmicEncoder(num_bits=8, base=2.0, feature_wise=True)`
- `num_bits`: Number of bits for encoding
- `base`: Logarithm base
- `feature_wise`: If True, compute ranges per feature

### Node Classes

All nodes inherit from `BaseNode(input_dim, output_dim, **kwargs)`.

#### `LinearLUTNode(input_dim=[n], output_dim=[1], init_fn=None)`
Simple linear transformation-based LUT with sigmoid activation.
- `input_dim`: Input dimensions as list (e.g., `[4]` for 4 inputs)
- `output_dim`: Output dimensions as list (e.g., `[1]` for single output)
- `init_fn`: Optional weight initialization function

#### `PolyLUTNode(input_dim=[n], output_dim=[1], degree=2, init_fn=None)`
Multivariate polynomial approximation up to degree D.
- `degree`: Maximum polynomial degree

#### `NeuralLUTNode(input_dim=[n], output_dim=[1], hidden_width=16, depth=4, skip_interval=2)`
Small MLP embedded in a LUT with skip connections.
- `hidden_width`: Width of hidden layers
- `depth`: Number of MLP layers
- `skip_interval`: Interval for skip connections (0 = no skips)

#### `DWNNode(input_dim=[n], output_dim=[1], hidden_size=32, num_layers=2)`
Deep Weight Networks - generates LUT weights through a small network.
- `hidden_size`: Hidden layer size in weight generator
- `num_layers`: Depth of weight generation network

#### `ProbabilisticNode(input_dim=[n], output_dim=[1], num_samples=10, temperature=1.0)`
Probabilistic LUT with uncertainty quantification via sampling.
- `num_samples`: Number of Monte Carlo samples
- `temperature`: Sampling temperature

#### `FourierNode(input_dim=[n], output_dim=[1], num_frequencies=8, use_cuda=True)`
Fourier basis approximation with optional CUDA acceleration.
- `num_frequencies`: Number of Fourier frequency components
- `use_cuda`: Enable CUDA kernel if available

#### `HybridNode(input_dim=[n], output_dim=[1], num_frequencies=8, mlp_hidden=16, use_cuda=True)`
Combines Fourier basis with learned MLP components.
- `num_frequencies`: Number of Fourier components
- `mlp_hidden`: Hidden size for MLP component
- `use_cuda`: Enable CUDA kernel if available

#### `GradientStabilizedNode(input_dim=[n], output_dim=[1], gradient_scale=1.0, use_cuda=True)`
Gradient-stabilized training with binary thresholding and smooth gradients.
- `gradient_scale`: Scaling factor for gradient magnitude
- `use_cuda`: Enable CUDA kernel if available
- **Forward**: Binary thresholding at 0.5 (DWN-style)
- **Backward**: Scaled gradients with input-dependent bounds for stable training

### Layer Classes

All layers inherit from `BaseLUTLayer(input_size, output_size, node_type, n, node_kwargs)`.

**Common Parameters:**
- `input_size`: Number of input features
- `output_size`: Number of output features (= number of LUT nodes)
- `node_type`: Node class (not string, pass the actual class)
- `n`: Number of inputs per LUT node
- `node_kwargs`: Dict of additional arguments passed to node constructor (e.g., `{'input_dim': [n], 'output_dim': [1], 'degree': 2}`)

#### `RandomLayer(..., seed=42)`
Random fixed connectivity pattern with uniform input distribution.
- `seed`: Random seed for reproducible mapping

#### `LearnableLayer(..., temperature=1.0, hard=False)`
Learns optimal input-to-node connections via Gumbel-softmax.
- `temperature`: Gumbel-softmax temperature (lower = harder selection)
- `hard`: Use hard selection in forward pass

#### `GroupedLayer(..., num_groups, group_strategy='sequential')`
Groups inputs semantically before connecting to nodes.
- `num_groups`: Number of input groups
- `group_strategy`: How to form groups ('sequential', 'random', etc.)

#### `ResidualLayer(..., base_layer_type, residual_weight=0.5)`
Adds residual/skip connections around a base LUT layer.
- `base_layer_type`: Base layer class (e.g., RandomLayer)
- `residual_weight`: Weighting for skip connection (0-1)


## Component Registry
Use the registry utilities to list or fetch available encoders, nodes, and layers by name.

DiffLUT uses a registration system for easy component discovery:

```python
from difflut.registry import get_registered_nodes, get_node_class
from difflut.registry import get_registered_layers, get_layer_class
from difflut.registry import get_registered_encoders, get_encoder_class

# List all available components
print(get_registered_nodes())
# ['linear_lut', 'polylut', 'neurallut', 'dwn', 'probabilistic', 'fourier', 'hybrid', 'gradient_stabilized']

print(get_registered_layers())
# ['random', 'learnable', 'grouped', 'residual']

print(get_registered_encoders())
# ['thermometer', 'gaussian', 'distributive', 'gray', 'onehot', 'binary', 'signmagnitude', 'logarithmic']

# Get a component class by its registered name
NodeClass = get_node_class('linear_lut')
node = NodeClass(input_dim=[4], output_dim=[1])

# Or import directly
from difflut.nodes import LinearLUTNode
node = LinearLUTNode(input_dim=[4], output_dim=[1])
```


## Tips
Start with LinearLUTNode and RandomLayer. Use 4-8 encoder bits. Prefer wider layers. Fit encoders before use. Enable CUDA for speed.

1. **Start Simple**: Begin with `LinearLUTNode` and `RandomLayer` for baseline performance
2. **Encoder Bits**: Use 4-8 bits for most applications; more bits = more LUT inputs needed
3. **LUT Input Size (n)**: Start with n=4 or n=6; larger n means exponentially more parameters (2^n LUT entries)
4. **Layer Width**: Use wider layers (more nodes) rather than deeper networks for better hardware efficiency
5. **Input/Output Dims**: Always specify as lists, e.g., `input_dim=[6]` not `input_dim=6`
6. **Regularization**: Add smoothness or sparsity regularization for better hardware efficiency
7. **CUDA**: Enable CUDA for nodes that support it (Fourier, Hybrid, GradientStabilized) for faster training
8. **Encoder Fitting**: Always call `encoder.fit(data)` before using `encoder.encode()` to set value ranges
9. **Gradient Stability**: Use `GradientStabilizedNode` for training stability with binary inputs


## Troubleshooting
If training is slow, check CUDA and use simpler nodes. For low accuracy, try more expressive nodes or wider layers.

**Q: Training is very slow**
- Check if CUDA is available: `torch.cuda.is_available()`
- Use simpler node types like `LinearLUTNode` instead of `NeuralLUTNode`
- Reduce `n` (number of LUT inputs) - exponential impact on compute

**Q: Poor accuracy compared to standard neural networks**
- Try different node types (DWN for more expressiveness)
- Increase layer width (more LUT nodes per layer)



## Package Structure

DiffLUT is organized as a self-contained Python package:

```
difflut/                # Top-level package directory
├── LICENSE             # License file
├── README.md           # Project documentation
├── __init__.py         # Package marker
├── difflut/            # Main library code
│   ├── __init__.py
│   ├── encoder/        # Input encoders (thermometer, gray, etc.)
│   ├── layers/         # Layer types (random, learnable, etc.)
│   ├── nodes/          # LUT node types and CUDA kernels
│   ├── registry.py     # Component registration utilities
│   └── utils/          # Utility functions (FPGA export, regularizers)
├── examples/           # Example scripts and notebooks
├── pyproject.toml      # Build system config
└── setup.py            # Installation script
```


### Installation as Package

The library can be installed independently from the parent research framework:

```bash
# From parent directory
pip install -e difflut/

# Or from within difflut directory
cd difflut
pip install -e .
```

This makes `difflut` available system-wide or in your virtual environment.


## Distribution and Publishing

### Building Distribution Packages

```bash
cd difflut/
python -m build
```

This creates:
- `dist/difflut-1.1.0.tar.gz` - Source distribution
- `dist/difflut-1.1.0-*.whl` - Wheel distribution (if applicable)

Note: CUDA extensions are compiled during installation, not during wheel building.

### Installing from Source Distribution

```bash
pip install difflut-1.1.0.tar.gz
```


## Contributing
Extend BaseNode, BaseLUTLayer, or BaseEncoder and register your component. See existing code for examples.

To add new components:

1. **New Node**: Extend `BaseNode` and use `@register_node` decorator
2. **New Layer**: Extend `BaseLUTLayer` and use `@register_layer` decorator
3. **New Encoder**: Extend `BaseEncoder` and use `@register_encoder` decorator

See existing implementations in `nodes/`, `layers/`, and `encoder/` for examples.

### Development Workflow

```bash
# Install in development mode with dev tools
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black difflut/
isort difflut/

# Lint code
flake8 difflut/
```


## Related Projects
Part of the DiffLUT Research Framework (see project repository).

This library is part of the larger **DiffLUT Research Framework**:
- **Parent Project**: Full experiment pipeline with Hydra configs, SLURM scripts
- **Repository**: https://gitlab.ethz.ch/disco-students/hs25/difflut
- **Experiments**: See `experiments/` directory in parent project


## Citation

If you use DiffLUT in your research, please cite:

```bibtex
@software{difflut2025,
  title={DiffLUT: Differentiable LUT Networks for Efficient FPGA Deployment},
  author={B\"uhrer, Simon Jonas},
  year={2025},
  institution={ETH Zurich, Distributed Computing Group},
  url={https://gitlab.ethz.ch/disco-students/hs25/difflut}
}
```


## License

MIT License - See LICENSE file for details.


## Contact

- **Author**: Simon Jonas Bührer
- **Email**: sbuehrer@ethz.ch
- **Institution**: ETH Zurich, Distributed Computing Group
- **Issues**: https://gitlab.ethz.ch/disco-students/hs25/difflut/-/issues
