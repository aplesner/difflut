# Developer Guide

This guide is for developers who want to extend DiffLUT, contribute new components, or deploy models to production.

## Who This Is For

- **Researchers** creating new node types or encoders
- **Contributors** improving the library
- **DevOps/ML Engineers** packaging and deploying DiffLUT
- **Advanced Users** building custom components for their use cases

## Overview

### Architecture

DiffLUT is built around **extensible base classes**:

- **BaseEncoder** - Extend to create new input encoders
- **BaseNode** - Extend to create new LUT node types
- **BaseLUTLayer** - Extend to create new layer connectivity patterns

All components are registered for easy discovery and configuration-driven building.

### Directory Structure

```
difflut/
├── encoder/
│   ├── base_encoder.py       # Extend to create encoders
│   ├── thermometer.py         # Example encoder
│   └── advanced_encoders.py   # More encoders
├── nodes/
│   ├── base_node.py           # Extend to create nodes
│   ├── linear_lut_node.py     # Simple LUT implementation
│   ├── polylut_node.py        # Polynomial LUT
│   ├── neural_lut_node.py     # Neural LUT
│   └── cuda/                  # CUDA kernels
├── layers/
│   ├── base_layer.py          # Extend to create layers
│   ├── random_layer.py        # Random connectivity
│   └── learnable_layer.py     # Learnable connectivity
├── registry.py                # Component registration
└── utils/                     # Utilities (FPGA export, etc.)
```

## Quick Links

### For Creating Components

- **[Creating Components Guide](DEVELOPER_GUIDE/creating_components.md)** - How to implement new nodes, encoders, and layers
  - Custom node example with registration
  - Custom encoder implementation
  - Custom layer connectivity
  - Testing your components

### For Packaging & Deployment

- **[Packaging & Distribution Guide](DEVELOPER_GUIDE/packaging.md)** - Build, package, and publish DiffLUT
  - Building source/wheel distributions
  - CUDA kernel compilation
  - Publishing to PyPI
  - Docker/Apptainer containers
  - Deployment best practices

### For Contributing

- **[Contributing Guide](DEVELOPER_GUIDE/contributing.md)** - Development setup and contribution guidelines
  - Setting up development environment
  - Running tests
  - Code style and conventions
  - Creating pull requests
  - Getting help

## Common Development Tasks

### Add a New Node Type

1. Create file `difflut/nodes/my_custom_node.py`
2. Extend `BaseNode` with your computation
3. Use `@register_node` decorator
4. Add tests in `tests/`

See [Creating Components](DEVELOPER_GUIDE/creating_components.md#custom-nodes) for detailed examples.

### Add a New Encoder

1. Create file `difflut/encoder/my_custom_encoder.py`
2. Extend `BaseEncoder`
3. Use `@register_encoder` decorator
4. Add tests

See [Creating Components](DEVELOPER_GUIDE/creating_components.md#custom-encoders) for examples.

### Add a New Layer Type

1. Create file `difflut/layers/my_custom_layer.py`
2. Extend `BaseLUTLayer`
3. Use `@register_layer` decorator
4. Add tests

See [Creating Components](DEVELOPER_GUIDE/creating_components.md#custom-layers) for examples.

### Build and Test

```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Build distributions
python -m build

# View documentation
cd docs/
```

### Add CUDA Support

1. Create CUDA kernel in `difflut/nodes/cuda/`
2. Add kernel launch code in node implementation
3. Add CPU fallback
4. Update `setup.py` for compilation

See [Creating Components](DEVELOPER_GUIDE/creating_components.md#cuda-support) for details.

## Development Workflow

### 1. Set Up Development Environment

```bash
git clone https://gitlab.ethz.ch/disco-students/hs25/difflut.git
cd difflut/difflut
pip install -e .
pip install -e ".[dev]"  # Install dev dependencies
```

### 2. Create Feature Branch

```bash
git checkout -b feature/my-new-component
```

### 3. Implement Your Component

See [Creating Components](DEVELOPER_GUIDE/creating_components.md) for detailed examples.

### 4. Write Tests

Add tests in `tests/test_my_component.py`:

```python
import torch
from difflut.nodes import MyCustomNode

def test_my_custom_node():
    node = MyCustomNode(input_dim=[4], output_dim=[1])
    x = torch.randint(0, 2, (32, 4))
    output = node(x)
    assert output.shape == (32, 1)
```

### 5. Run Tests

```bash
python -m pytest tests/
```

### 6. Commit and Push

```bash
git add -A
git commit -m "Add MyCustomNode with registration"
git push origin feature/my-new-component
```

### 7. Create Pull Request

See [Contributing Guide](DEVELOPER_GUIDE/contributing.md) for PR guidelines.

## Key Concepts

### Component Registration

All DiffLUT components use a decorator-based registration system:

```python
from difflut import register_node, register_encoder, register_layer

@register_node('my_custom_node')
class MyCustomNode(BaseNode):
    pass

@register_encoder('my_custom_encoder')
class MyCustomEncoder(BaseEncoder):
    pass

@register_layer('my_custom_layer')
class MyCustomLayer(BaseLUTLayer):
    pass
```

Once registered, components are discoverable:

```python
from difflut.registry import get_node_class
NodeClass = get_node_class('my_custom_node')
```

### CUDA Extensions

CUDA kernels are compiled automatically during installation if CUDA is available:

```bash
# Compilation happens automatically
pip install -e .

# Or explicitly:
export CUDA_HOME=/usr/local/cuda-11.8
pip install -e .
```

CPU fallbacks are required for all CUDA-accelerated nodes.

### Testing

Use pytest for all tests:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_all_nodes_training.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=difflut tests/
```

## Project Structure for Contributors

```
difflut/
├── difflut/
│   ├── encoder/          # Encoders - extend BaseEncoder
│   ├── nodes/            # Nodes - extend BaseNode
│   ├── layers/           # Layers - extend BaseLUTLayer
│   ├── registry.py       # Component registration
│   └── utils/            # Utilities
├── tests/                # Test suite - add tests here
├── examples/             # Examples - add demos here
├── docs/
│   ├── DEVELOPER_GUIDE/
│   │   ├── creating_components.md
│   │   ├── packaging.md
│   │   └── contributing.md
│   └── ...
├── setup.py              # Installation & CUDA compilation
└── pyproject.toml        # Build configuration
```

## Common Patterns

### Adding a Simple Node

```python
from difflut.nodes import BaseNode
from difflut import register_node
import torch
import torch.nn as nn

@register_node('simple_node')
class SimpleNode(BaseNode):
    def __init__(self, input_dim=None, output_dim=None, **kwargs):
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)
        # Initialization
        self.weights = nn.Parameter(
            torch.randn(2**self.num_inputs, self.num_outputs)
        )
    
    def forward_train(self, x):
        """Training forward pass"""
        indices = self._binary_to_index(x)
        return self.weights[indices]
    
    def forward_eval(self, x):
        """Evaluation forward pass"""
        return self.forward_train(x)
```

### Adding CUDA Support

```python
import torch
from difflut import register_node

@register_node('cuda_accelerated_node')
class CUDAAcceleratedNode(BaseNode):
    def forward_train(self, x):
        if torch.cuda.is_available() and x.is_cuda:
            # Use CUDA kernel
            return self._forward_cuda(x)
        else:
            # CPU fallback
            return self._forward_cpu(x)
    
    def _forward_cuda(self, x):
        # Load CUDA extension
        try:
            from difflut.nodes.cuda import my_cuda_kernel
            return my_cuda_kernel(x, self.weights)
        except ImportError:
            # Fallback if CUDA not available
            return self._forward_cpu(x)
    
    def _forward_cpu(self, x):
        # Pure PyTorch implementation
        indices = self._binary_to_index(x)
        return self.weights[indices]
```

## Resources

- **[Creating Components](DEVELOPER_GUIDE/creating_components.md)** - Detailed implementation guide
- **[Packaging & Distribution](DEVELOPER_GUIDE/packaging.md)** - Build and publish guide
- **[Contributing](DEVELOPER_GUIDE/contributing.md)** - Contribution guidelines
- **Source Code** - `difflut/` directory
- **Tests** - `tests/` directory
- **Examples** - `examples/` directory

## Getting Help

- **Issues**: https://gitlab.ethz.ch/disco-students/hs25/difflut/-/issues
- **Discussions**: Check existing issues and documentation
- **Email**: sbuehrer@ethz.ch

## Next Steps

1. Read [Creating Components](DEVELOPER_GUIDE/creating_components.md) to understand the architecture
2. Look at existing implementations as examples
3. Create your component following the patterns
4. Write tests using provided test utilities
5. Follow [Contributing](DEVELOPER_GUIDE/contributing.md) guidelines for PRs
