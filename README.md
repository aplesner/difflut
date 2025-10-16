
# DiffLUT Library
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)]()

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


## Related Projects
Part of the DiffLUT Research Framework.

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
