
# DiffLUT Library
![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)]()

DiffLUT is a modular PyTorch library for differentiable Look-Up Table (LUT) neural networks, designed for efficient FPGA deployment and research. It provides encoders, LUT nodes, flexible layers, and CUDA acceleration.

## ✨ Features

- **Modular Architecture**: Encoders, LUT nodes, and layers as composable building blocks
- **Multiple Node Types**: LinearLUT, PolyLUT, NeuralLUT, DWN, Probabilistic, Fourier, Hybrid, Gradient-Stabilized
- **Input Encoders**: Thermometer, Gaussian, Gray, OneHot, Binary, Logarithmic, and more
- **Flexible Layers**: Random, Learnable, Grouped, and Residual connectivity patterns
- **CUDA Acceleration**: Optional GPU support for compute-intensive nodes
- **FPGA Export**: Tools for deploying trained networks to FPGAs
- **Component Registry**: Easy discovery and instantiation of components

## 🚀 Quick Links

### For Users
- **[Installation Guide](docs/INSTALLATION.md)** - Setup and requirements
- **[Quick Start](docs/QUICK_START.md)** - Get running in 5 minutes
- **[User Guide](docs/USER_GUIDE.md)** - Learn the library components and patterns
  - [Components Guide](docs/USER_GUIDE/components.md) - Encoders, nodes, and layers
  - [Registry & Pipelines](docs/USER_GUIDE/registry_pipeline.md) - Component discovery and pipeline building

### For Developers
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Extend and contribute to DiffLUT
  - [Creating Components](docs/DEVELOPER_GUIDE/creating_components.md) - Implement custom nodes/encoders/layers
  - [Packaging & Distribution](docs/DEVELOPER_GUIDE/packaging.md) - Build and publish DiffLUT
  - [Contributing](docs/DEVELOPER_GUIDE/contributing.md) - Development setup and guidelines

## 📦 Package Structure

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
│   │   └── utils/             # initializers and regularizer
│   └── utils/                 # Utility functions (FPGA export)
├── examples/                  # Example scripts and notebooks
└── tests/                     # Test suite
```

## 📚 Citation

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

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 👤 Contact

- **Author**: Simon Jonas Bührer
- **Email**: sbuehrer@ethz.ch
- **Institution**: ETH Zurich, Distributed Computing Group
- **Issues**: https://gitlab.ethz.ch/disco-students/hs25/difflut/-/issues

## 🔗 Related Projects

Part of the **DiffLUT Research Framework**:
- **Parent Project**: Full experiment pipeline with Hydra configs and SLURM scripts
- **Repository**: https://gitlab.ethz.ch/disco-students/hs25/difflut
