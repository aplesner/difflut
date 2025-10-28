# Packaging & Distribution Guide

Learn how to build, package, and distribute DiffLUT.

## Overview

DiffLUT supports multiple distribution formats:

- **Source Distribution** (`.tar.gz`) - For development and source inspection
- **Wheel Distribution** (`.whl`) - For binary installation
- **PyPI** - For public package repository
- **Docker/Singularity** - For containerized environments

## Prerequisites

```bash
# Install build tools
pip install build twine wheel setuptools

# For PyPI publishing (optional)
pip install keyring
```

## Building Distributions

### Build Both Source and Wheel

```bash
cd difflut/
python -m build
```

This creates:
- `dist/difflut-*.tar.gz` - Source distribution
- `dist/difflut-*.whl` - Wheel distribution

### Build Only Source Distribution

```bash
python -m build --sdist
```

### Build Only Wheel

```bash
python -m build --wheel
```

### Verbose Build

```bash
python -m build -v
```

## CUDA Compilation

### Automatic CUDA Detection

CUDA extensions are compiled automatically if CUDA is available:

```bash
# CUDA detected automatically during build
python -m build
```

### Specifying CUDA Version

```bash
# Set CUDA_HOME before building
export CUDA_HOME=/usr/local/cuda-11.8
python -m build
```

### CPU-Only Build

```bash
# Build without CUDA support
export CUDA_HOME=""
python -m build
```

### Verifying CUDA Compilation

```python
# Check if CUDA extensions loaded
try:
    from difflut.nodes.cuda import fourier_cuda
    print("✓ CUDA extensions loaded")
except ImportError:
    print("✗ CPU fallback (CUDA not available)")
```

## Installation Methods

### From Source Distribution

```bash
# Build source distribution
python -m build --sdist

# Install from .tar.gz
pip install dist/difflut-*.tar.gz
```

CUDA extensions are compiled during installation.

### From Wheel Distribution

```bash
# Build wheel
python -m build --wheel

# Install from .whl
pip install dist/difflut-*.whl
```

**Note**: Wheels include pre-compiled extensions for the build platform.

### Editable Installation (Development)

```bash
# Install in editable mode for development
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

CUDA extensions are compiled during installation.

### From PyPI (When Published)

```bash
pip install difflut
```

## Publishing to PyPI

### 1. Prepare Release

```bash
# Update version in setup.py
# Update CHANGELOG
# Update docs

git tag v1.1.0
git push --tags
```

### 2. Build Distributions

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build
python -m build
```

### 3. Create PyPI Account

Visit https://pypi.org/account/register/

### 4. Configure Credentials

Create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-AgEIcHlwaS5vcmc...

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-...
```

Or use keyring:

```bash
keyring set https://upload.pypi.org/legacy/ __token__
```

### 5. Upload to TestPyPI (Recommended First)

```bash
twine upload --repository testpypi dist/*
```

### 6. Test Installation

```bash
pip install -i https://test.pypi.org/simple/ difflut
```

### 7. Upload to PyPI

```bash
twine upload dist/*
```

## Building Containers

### Docker Build

Create `Dockerfile`:

```dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir numpy scipy

# Copy and install DiffLUT
COPY . .
RUN pip install -e .

# Set entry point
CMD ["python"]
```

Build:

```bash
docker build -t difflut:latest .
docker run --gpus all -it difflut:latest
```

### Singularity Container

Create `Singularity.def`:

```singularity
Bootstrap: docker
From: pytorch/pytorch:2.0-cuda11.8-cudnn8-runtime

%post
    apt-get update && apt-get install -y git
    pip install --no-cache-dir numpy scipy
    
    cd /tmp
    git clone https://gitlab.ethz.ch/disco-students/hs25/difflut.git
    cd difflut/difflut
    pip install -e .

%environment
    export CUDA_VISIBLE_DEVICES=0

%runscript
    python "$@"
```

Build and run:

```bash
singularity build difflut.sif Singularity.def
singularity run --nv difflut.sif python -c "import difflut; print('✓ DiffLUT loaded')"
```

## Version Management

### Semantic Versioning

Follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

```python
# setup.py or pyproject.toml
version = "1.1.0"
```

### Version in Code

```python
# difflut/__init__.py
__version__ = "1.1.0"
```

Check version:

```python
import difflut
print(difflut.__version__)
```

## Package Metadata

### setup.py Configuration

```python
from setuptools import setup, find_packages

setup(
    name="difflut",
    version="1.1.0",
    description="Differentiable LUT networks for efficient FPGA deployment",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Simon Jonas Bührer",
    author_email="sbuehrer@ethz.ch",
    url="https://gitlab.ethz.ch/disco-students/hs25/difflut",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9",
        "numpy",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
```

### pyproject.toml Configuration

```toml
[project]
name = "difflut"
version = "1.1.0"
description = "Differentiable LUT networks for efficient FPGA deployment"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Simon Jonas Bührer", email = "sbuehrer@ethz.ch"}
]
requires-python = ">=3.7"
dependencies = [
    "torch>=1.9",
    "numpy",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "black",
    "flake8",
]

[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"
```

## Deployment Best Practices

### Environment Parity

Ensure consistency across environments:

```bash
# Export environment
pip freeze > requirements.txt

# Reproduce in new environment
pip install -r requirements.txt
```

### GPU Memory Management

```python
import torch

# Check available memory
print(torch.cuda.get_device_properties(0).total_memory)

# Clear cache if needed
torch.cuda.empty_cache()
```

### Batch Processing

For large-scale inference:

```python
import torch
from torch.utils.data import DataLoader

def predict_batch(model, dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    
    with torch.no_grad():
        for batch in loader:
            output = model(batch.cuda())
            predictions.append(output.cpu())
    
    return torch.cat(predictions)
```

### Model Serialization

```python
import torch

# Save model
checkpoint = {
    'model': model.state_dict(),
    'config': config,
    'version': '1.1.0'
}
torch.save(checkpoint, 'model.pt')

# Load model
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['model'])
print(f"Loaded model version {checkpoint['version']}")
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run tests
      run: pytest tests/
    
    - name: Build distributions
      run: python -m build
    
    - name: Upload to PyPI (tagged releases only)
      if: startsWith(github.ref, 'refs/tags/')
      run: twine upload dist/*
```

### GitLab CI Example

```yaml
stages:
  - build
  - test
  - publish

build:
  stage: build
  script:
    - python -m build

test:
  stage: test
  script:
    - pip install -e ".[dev]"
    - pytest tests/

publish:
  stage: publish
  script:
    - twine upload dist/*
  only:
    - tags
```

## Release Checklist

- [ ] Update version in `setup.py` and `__init__.py`
- [ ] Update `CHANGELOG.md`
- [ ] Update documentation
- [ ] Run full test suite
- [ ] Build distributions locally
- [ ] Test installation from distribution
- [ ] Test on multiple Python versions
- [ ] Create git tag
- [ ] Upload to TestPyPI
- [ ] Test installation from TestPyPI
- [ ] Upload to PyPI
- [ ] Create release notes
- [ ] Announce release

## Troubleshooting

### CUDA Compilation Fails

```bash
# Check CUDA_HOME
echo $CUDA_HOME

# Use verbose output
python -m build -v

# Try CPU-only build
export CUDA_HOME=""
pip install -e .
```

### Version Mismatch

```bash
# Check installed version
python -c "import difflut; print(difflut.__version__)"

# Reinstall fresh
pip uninstall difflut -y
pip install -e .
```

### Build Cache Issues

```bash
# Clean build artifacts
rm -rf build/ dist/ *.egg-info __pycache__

# Rebuild
python -m build
```

## Resources

- **Build Documentation**: https://packaging.python.org/
- **PyPI Help**: https://pypi.org/help/
- **Setuptools**: https://setuptools.pypa.io/
- **Twine**: https://twine.readthedocs.io/
- **Docker Docs**: https://docs.docker.com/
- **Singularity Docs**: https://sylabs.io/guides/3.0/user-guide/

## Next Steps

- [Developer Guide](../DEVELOPER_GUIDE.md) - Overview of development
- [Contributing](contributing.md) - Contribution guidelines
- [Creating Components](creating_components.md) - Implement new features
