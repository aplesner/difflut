
# DiffLUT Library
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)]()

DiffLUT is a modular PyTorch library for differentiable Look-Up Table (LUT) neural networks, designed for efficient FPGA deployment and research. It provides encoders, LUT nodes, flexible layers, and CUDA acceleration.

## Features

- **Modular Architecture**: Encoders, LUT nodes, and layers as composable building blocks
- **Multiple Node Types**: LinearLUT, PolyLUT, NeuralLUT, DWN, DWNStable, Probabilistic, Fourier, Hybrid
- **Input Encoders**: Thermometer, Gaussian, Distributive, Gray, OneHot, Binary, SignMagnitude, Logarithmic
- **Flexible Layers**: Random and Learnable connectivity patterns
- **CUDA Acceleration**: Optional GPU support for compute-intensive nodes
- **FPGA Export**: Tools for deploying trained networks to FPGAs
- **Component Registry**: Easy discovery and instantiation of components

## 📖 Documentation

**[📋 Documentation Structure Overview](docs/DOCUMENTATION_STRUCTURE.md)** - Understand how documentation is organized

### Getting Started
1. **[Installation Guide](docs/INSTALLATION.md)** - Setup, dependencies, and CUDA configuration
2. **[Quick Start](docs/QUICK_START.md)** - Build your first LUT network in 5 minutes

### User Documentation
- **[User Guide](docs/USER_GUIDE.md)** - Overview of all user-facing features
  - **[Components Guide](docs/USER_GUIDE/components.md)** - Encoders, nodes, layers, and initializers
    - [Encoders](docs/USER_GUIDE/components/encoders.md) - Available input encoders
    - [Nodes](docs/USER_GUIDE/components/nodes.md) - Available LUT node types
    - [Layers](docs/USER_GUIDE/components/layers.md) - Layer connectivity patterns
    - [Models](docs/USER_GUIDE/components/models.md) - Using and configuring models
  - **[Registry & Pipelines](docs/USER_GUIDE/registry_pipeline.md)** - Component discovery and configuration-driven building
  - **[Export Guide](docs/USER_GUIDE/export_guide.md)** - Export models for FPGA and C deployment

### Developer Documentation
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Architecture overview and contributing
  - **[Creating Components](docs/DEVELOPER_GUIDE/creating_components.md)** - Overview of component architecture
    - [Creating Encoders](docs/DEVELOPER_GUIDE/creating_components/creating_encoders.md) - Implement custom encoders
    - [Creating Nodes](docs/DEVELOPER_GUIDE/creating_components/creating_nodes.md) - Implement custom LUT nodes
    - [Creating Layers](docs/DEVELOPER_GUIDE/creating_components/creating_layers.md) - Implement custom layers
    - [Creating Models](docs/DEVELOPER_GUIDE/creating_components/creating_models.md) - Implement custom model architectures
    - [Creating Blocks](docs/DEVELOPER_GUIDE/creating_components/creating_blocks.md) - Compose components into blocks *(future)*
  - **[Testing](docs/DEVELOPER_GUIDE/tests.md)** - Test framework and guidelines
  - **[Packaging & Distribution](docs/DEVELOPER_GUIDE/packaging.md)** - Build, test, and publish
  - **[Contributing](docs/DEVELOPER_GUIDE/contributing.md)** - Development setup and workflow
  - **[GitHub Workflow](docs/DEVELOPER_GUIDE/github_guide.md)** - GitHub-specific practices
  - **[Apptainer Containers](docs/DEVELOPER_GUIDE/apptainer.md)** - Container setup for development

## Package Structure

DiffLUT is organized as a self-contained Python package:

```
difflut/                       # Top-level package directory
├── README.md                  # This file
├── LICENSE                    # MIT License
├── setup.py                   # Installation script
├── pyproject.toml             # Build system config
├── docs/                      # Documentation
├── difflut/                   # Main library code
│   ├── __init__.py
│   ├── registry.py            # Component registration utilities
│   ├── encoder/               # Input encoders
│   ├── layers/                # Layer types
│   ├── nodes/                 # LUT node implementations
│   │   └── utils/             # initializers and regularizers
│   ├── models/                # Model Zoo (ready-to-use architectures)
│   └── utils/                 # Utility functions (FPGA export)
├── examples/                  # Example scripts and notebooks
└── tests/                     # Test suite
```

## 📚 Citation

If you use DiffLUT in your research, please cite:

```bibtex
@software{difflut2025,
  title={DiffLUT: Differentiable Lookup Table Networks},
  author={B\"uhrer, Simon Jonas and Plesner, Andreas and Aczel, Till},
  year={2025},
  institution={ETH Zurich, Distributed Computing Group},
  url={https://github.com/aplesner/difflut}
}
```

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 👤 Contact

- **Authors**: Simon Jonas Bührer, Andreas Plesner, and Till Aczel
- **Emails**: {sbuehrer,aplesner,taczel}@ethz.ch
- **Institution**: ETH Zurich, Distributed Computing Group
- **Issues**: https://github.com/aplesner/difflut/-/issues

## 🔗 Related Projects

Part of the **DiffLUT Research Framework**:
- **Parent Project**: Full experiment pipeline with Hydra configs and SLURM scripts
