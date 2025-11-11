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

- **[Creating Components Guide](DEVELOPER_GUIDE/creating_components.md)**
  - How to implement new nodes, encoders, and layers
  - Custom node example with registration
  - Custom encoder implementation
  - Custom layer implementation
  - Custom initializer implementation
  - Custom regularizer implementation

### For Testing Components

- **[Test Components Guide](DEVELOPER_GUIDE/tests.md)**
    - How to use tests
    - How to implemnt your own test
    - Github CI test workflow

### For Packaging & Deployment

- **[Packaging & Distribution Guide](DEVELOPER_GUIDE/packaging.md)** - Build, package, and publish DiffLUT
  - Building source/wheel distributions
  - CUDA kernel compilation
  - Publishing to PyPI
  - Deployment best practices

### For Contributing

- **[Contributing Guide](DEVELOPER_GUIDE/contributing.md)** - Development setup and contribution guidelines
  - Setting up development environment
  - Running tests
  - Code style and conventions
  - Creating pull requests
  - Getting help


## Next Steps
1. Read [Creating Components](DEVELOPER_GUIDE/creating_components.md) to understand the architecture
2. Look at existing implementations as examples
3. Create your component following the patterns
4. Write tests using provided test utilities
5. Follow [Contributing](DEVELOPER_GUIDE/contributing.md) guidelines for PRs
