# Installation Guide

This guide covers installing DiffLUT for production use, development, and different hardware configurations.

## Requirements

- **Python**: 3.7 or higher
- **PyTorch**: 1.9 or higher
- **NumPy**: 1.19 or higher
- **CUDA Toolkit** (optional): Required for GPU-accelerated CUDA nodes

### GPU-Accelerated Nodes
The following nodes have optional CUDA acceleration:
- Fourier Node (`fourier_cuda`)
- Hybrid Node (`hybrid_cuda`)
- DWN Node (`efd_cuda`)
- DWN Stable Node (`dwn_stable_cuda`)
- Probabilistic Node (`probabilistic_cuda`)

CPU fallback implementations are available for all nodes.

---

## Installation Options

### 📦 From PyPI (Recommended)

#### Standard Installation (GPU Support, Default)
Installs with CUDA extensions if CUDA is available:

```bash
pip install difflut
```

#### GPU Installation (Explicit)
Force GPU build with CUDA support:

```bash
pip install difflut[gpu]
```

#### CPU-Only Installation
Skip CUDA build for CPU-only systems:

```bash
pip install difflut[cpu]
```

#### Development Installation
Install with development tools (testing, linting, formatting, documentation):

```bash
pip install difflut[dev]
```

Development dependencies include:
- `bump2version` - Version management
- `pytest`, `pytest-cov` - Testing framework
- `black`, `flake8`, `isort` - Code formatting and linting
- `sphinx`, `sphinx-rtd-theme` - Documentation generation
- `jupyter` - Notebook support
- `matplotlib`, `torchvision` - Visualization and datasets

---

### 🔧 From Source

#### Clone the Repository

```bash
git clone https://github.com/aplesner/difflut.git
cd difflut/difflut
```

#### Standard Installation (GPU)
```bash
pip install .
```

#### CPU-Only Installation
```bash
pip install .[cpu]
```

#### Development Installation (Editable)
Install in editable mode with development tools:

```bash
pip install -e .[dev]
```

This allows you to modify the source code and see changes immediately without reinstalling.

---

## GPU / CUDA Support

### Automatic CUDA Detection

DiffLUT automatically detects CUDA availability during installation:

1. **Checks for CUDA_HOME** environment variable
2. **Checks PyTorch CUDA** availability
3. **Compiles CUDA extensions** if available
4. **Falls back to CPU** if CUDA is not detected

### CUDA Extensions Built

When CUDA is available, these extensions are compiled:
- `dwn_stable_cuda` - DWN Stable Node acceleration
- `efd_cuda` - Efficient Feature Descriptor (DWN Node)
- `fourier_cuda` - Fourier Node acceleration
- `hybrid_cuda` - Hybrid Node acceleration
- `probabilistic_cuda` - Probabilistic Node acceleration
- `learnable_mapping_cuda` - Learnable Layer acceleration
- `mapping_cuda` - Random Layer acceleration

### Verifying CUDA Support

After installation, verify CUDA support:

```python
import torch
import difflut

# Check PyTorch CUDA availability
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"DiffLUT version: {difflut.__version__}")

# Try importing CUDA-accelerated nodes
try:
    from difflut.nodes import FourierNode, HybridNode, DWNStableNode
    print("✅ CUDA-accelerated nodes loaded successfully")
except ImportError as e:
    print(f"⚠️  CUDA nodes not available: {e}")
```

### Specifying CUDA Version

To compile against a specific CUDA version:

```bash
# Set CUDA_HOME before installation
export CUDA_HOME=/usr/local/cuda-11.8
pip install difflut

# Or for development
export CUDA_HOME=/usr/local/cuda-11.8
pip install -e .[dev]
```

---

## Combining Installation Options

You can combine multiple extras:

```bash
# Development + GPU
pip install difflut[dev,gpu]

# Development + CPU-only
pip install difflut[dev,cpu]

# All dependencies
pip install difflut[all]
```

---

## Troubleshooting

### CUDA Compilation Issues

If CUDA kernel compilation fails:

1. **Check CUDA Toolkit Installation**:
   ```bash
   nvcc --version
   ```

2. **Verify CUDA_HOME** is set correctly:
   ```bash
   echo $CUDA_HOME
   # Should point to CUDA installation (e.g., /usr/local/cuda or /usr/local/cuda-11.8)
   ```

3. **Check PyTorch CUDA compatibility**:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
   ```

4. **Install CPU-only version** as fallback:
   ```bash
   pip install difflut[cpu]
   ```

### Build Fails During Install

If you see compilation errors:

```bash
# Try installing without build isolation
pip install difflut --no-build-isolation

# Or force CPU-only build
pip install difflut[cpu]
```

### PyTorch Version Compatibility

Ensure PyTorch version compatibility:

```bash
# Check installed PyTorch version
python -c "import torch; print(torch.__version__)"

# Upgrade PyTorch if needed (CPU)
pip install --upgrade torch torchvision

# Upgrade PyTorch if needed (GPU, CUDA 11.8 example)
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Missing System Dependencies

**Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip build-essential
```

**macOS** (with Homebrew):
```bash
brew install python
xcode-select --install  # Install command line tools
```

### Import Errors After Installation

If you get import errors:

```bash
# Verify installation
pip show difflut

# Reinstall if needed
pip uninstall difflut
pip install difflut

# Check for conflicting packages
pip list | grep difflut
```

---

## Building Distribution Packages

For package maintainers and developers:

### Build Source and Wheel Distributions

```bash
# Install build tools
pip install build twine

# Build distributions
cd difflut
python -m build
```

This creates:
- `dist/difflut-*.tar.gz` - Source distribution
- `dist/difflut-*.whl` - Wheel distribution

**Note**: CUDA extensions are compiled during installation, not during wheel building.

### Upload to PyPI

```bash
# Test PyPI (recommended first)
twine upload --repository testpypi dist/*

# Production PyPI
twine upload dist/*
```

### Version Management

Use `bump2version` (included in dev dependencies) to manage versions:

```bash
# Install dev dependencies
pip install -e .[dev]

# Bump version
bump2version patch  # 1.1.2 → 1.1.3
bump2version minor  # 1.1.2 → 1.2.0
bump2version major  # 1.1.2 → 2.0.0
```

See `VERSION.md` for details.

---

## Container Support

A containerized environment is available via Apptainer:

```bash
# Run with GPU support
singularity run --nv containers/pytorch_universal_minimal.sif

# Run CPU-only
singularity run containers/pytorch_universal_minimal.sif
```

The `--nv` flag enables NVIDIA GPU support. See `containers/` directory for container definitions.

---

## Verification

Test your installation:

```python
import torch
import difflut
from difflut.encoder import ThermometerEncoder
from difflut.nodes import LinearLUTNode
from difflut.layers import RandomLayer

print(f"DiffLUT version: {difflut.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Create a simple model
encoder = ThermometerEncoder(num_bits=4)
layer = RandomLayer(
    input_size=32,
    output_size=16,
    node_type=LinearLUTNode,
    n=4,
    node_kwargs={'input_dim': [4], 'output_dim': [1]}
)

# Test forward pass
x = torch.randn(8, 32)
encoded = encoder(x)
output = layer(encoded)

print(f"✅ Installation successful! Output shape: {output.shape}")
```

---

## Quick Reference

| Installation Type | Command | Use Case |
|------------------|---------|----------|
| **Standard (GPU)** | `pip install difflut` | Default, with CUDA if available |
| **GPU Explicit** | `pip install difflut[gpu]` | Force GPU build |
| **CPU Only** | `pip install difflut[cpu]` | Skip CUDA build |
| **Development** | `pip install difflut[dev]` | Testing, linting, docs |
| **All Extras** | `pip install difflut[all]` | Everything |
| **From Source** | `pip install .` | Local development |
| **Editable** | `pip install -e .[dev]` | Active development |

---

## Next Steps

- 📖 [Quick Start](QUICK_START.md) - Build your first LUT network in 5 minutes
- 📚 [User Guide](USER_GUIDE.md) - Comprehensive usage examples
- 🔧 [Developer Guide](DEVELOPER_GUIDE.md) - Extend and contribute
- 💻 [Examples](../examples/) - Jupyter notebooks and scripts

---

## Getting Help

- **Documentation**: https://github.com/aplesner/difflut/tree/main/docs
- **Issues**: https://github.com/aplesner/difflut/issues
- **Email**: sbuehrer@ethz.ch
