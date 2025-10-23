# Installation Guide

This guide covers installing DiffLUT for development or use.

## Requirements

- **Python**: 3.7 or higher
- **PyTorch**: 1.9 or higher
- **CUDA** (optional): Required for GPU acceleration on CUDA-capable nodes (Fourier, Hybrid, Gradient-Stabilized)

## Basic Installation

### From Source (Development)

Clone the repository and install in editable mode:

```bash
git clone https://gitlab.ethz.ch/disco-students/hs25/difflut.git
cd difflut/difflut
pip install -e .
```

This installs DiffLUT with all dependencies and compiles CUDA extensions if CUDA is available on your system.

### From PyPI

Once published to PyPI:

```bash
pip install difflut
```

## GPU / CUDA Support

### Automatic CUDA Support

CUDA kernels for the following nodes are compiled automatically if CUDA is available:

- **Fourier Node**: `fourier_cuda`
- **Hybrid Node**: `hybrid_cuda`
- **Gradient-Stabilized Node**: `gradient_stabilized_cuda`
- **Probabilistic Node**: `probabilistic_cuda`
- **DWN Stable Node**: `dwn_stable_cuda`

If CUDA is not available, CPU fallback implementations are used automatically.

### Verifying CUDA Support

Check if CUDA extensions loaded successfully:

```python
import torch
import difflut

# Check PyTorch CUDA availability
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

# Check if CUDA-accelerated nodes loaded
from difflut.nodes import FourierNode, HybridNode
print("CUDA-accelerated nodes loaded successfully")
```

### Installing with Specific CUDA Version

If you need to compile against a specific CUDA version:

```bash
# Set CUDA_HOME environment variable before install
export CUDA_HOME=/usr/local/cuda-11.8
pip install -e .
```

## Troubleshooting Installation

### CUDA Compilation Issues

If CUDA kernel compilation fails:

1. **Check CUDA Toolkit Installation**:
   ```bash
   nvcc --version
   ```

2. **Verify CUDA_HOME** is set correctly:
   ```bash
   echo $CUDA_HOME
   # Should point to your CUDA installation directory
   ```

3. **Use CPU-only mode** if CUDA is unavailable:
   ```bash
   pip install -e . --no-build-isolation
   ```

### PyTorch Version Compatibility

If you encounter PyTorch version issues:

```bash
# Check installed PyTorch version
python -c "import torch; print(torch.__version__)"

# Upgrade PyTorch if needed
pip install --upgrade torch torchvision
```

### Missing Dependencies

If installation fails due to missing system dependencies:

- **Ubuntu/Debian**:
  ```bash
  sudo apt-get update
  sudo apt-get install python3-dev python3-pip
  ```

- **macOS** (with Homebrew):
  ```bash
  brew install python
  ```

## Building Distribution Packages

To build source and wheel distributions:

```bash
cd difflut
pip install build
python -m build
```

This creates:
- `dist/difflut-*.tar.gz` - Source distribution
- `dist/difflut-*.whl` - Wheel distribution (binary)

**Note**: CUDA extensions are compiled during installation, not during wheel building. Wheels are platform-specific.

## Installing from Source Distribution

```bash
pip install difflut-1.1.0.tar.gz
```

## Docker

A containerized environment is available via Singularity:

```bash
singularity run --nv pytorch_universal_minimal.sif
```

The `--nv` flag enables GPU support. See `containers/` for container definitions.

## Verification

Test your installation:

```python
import torch
import difflut
from difflut.encoder import ThermometerEncoder
from difflut.nodes import LinearLUTNode
from difflut.layers import RandomLayer

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

print(f"✓ Installation successful! Output shape: {output.shape}")
```

## Next Steps

- See [Quick Start](QUICK_START.md) to build your first LUT network
- Read [User Guide](USER_GUIDE.md) for comprehensive usage examples
