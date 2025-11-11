# Packaging & Distribution Guide

Quick guide for building and publishing DiffLUT distributions.

> **Note**: For installation instructions, see [`docs/INSTALLATION.md`](../INSTALLATION.md). For version management and release workflow, see [`.github/GITHUB_GUIDE.md`](../../.github/GITHUB_GUIDE.md). For container usage, see [`apptainer/README.md`](../../apptainer/README.md).

---

## Prerequisites

```bash
# Install build tools
pip install build twine

# For version bumping (required before releases)
pip install bump2version
```

---

## Building Distributions

### Build Source and Wheel Distributions

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build both .tar.gz and .whl
cd difflut/
python -m build

# Output:
# dist/difflut-1.3.2.tar.gz    (source distribution)
# dist/difflut-1.3.2-*.whl     (wheel distribution)
```

### Build Options

```bash
# Source distribution only
python -m build --sdist

# Wheel only
python -m build --wheel

# Verbose output (useful for debugging CUDA compilation)
python -m build -v
```

---

## CUDA Compilation

CUDA extensions are automatically compiled if CUDA is detected during build.

### Force CPU-Only Build

```bash
# Disable CUDA support
export CUDA_HOME=""
python -m build
```

### Specify CUDA Version

```bash
# Set CUDA_HOME before building
export CUDA_HOME=/usr/local/cuda-12.4
python -m build
```

### Verify CUDA Extensions

```python
# After installation, check if CUDA extensions loaded
python -c "
try:
    from difflut.nodes.cuda import dwn_stable_cuda
    print('✓ CUDA extensions available')
except ImportError:
    print('✗ CPU fallback (CUDA not available)')
"
```

---

## Publishing to PyPI

### 1. Bump Version (Required)

Before publishing, **always bump the version** using `bump2version`:

```bash
# For bug fixes
bump2version patch   # 1.3.2 → 1.3.3

# For new features
bump2version minor   # 1.3.2 → 1.4.0

# For breaking changes
bump2version major   # 1.3.2 → 2.0.0

# Push with tags
git push --follow-tags
```

See [`.github/GITHUB_GUIDE.md`](../../.github/GITHUB_GUIDE.md) for version management details.

### 2. Build Distributions

```bash
# Clean and build
rm -rf build/ dist/ *.egg-info
python -m build
```

### 3. Upload to TestPyPI (Recommended First)

```bash
# Test upload
twine upload --repository testpypi dist/*

# Test installation
pip install -i https://test.pypi.org/simple/ difflut
```

### 4. Upload to PyPI

```bash
# Production upload
twine upload dist/*
```

### Configure PyPI Credentials

Create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...

[testpypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcC...
```

Or use environment variables:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgEIcHlwaS5vcmc...
twine upload dist/*
```

---

## Local Installation Testing

### Install from Built Distributions

```bash
# From source distribution
pip install dist/difflut-1.3.2.tar.gz

# From wheel
pip install dist/difflut-1.3.2-*.whl

# Editable install (development)
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

### Verify Installation

```python
import difflut
print(f"DiffLUT version: {difflut.__version__}")

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## Release Checklist

Before publishing a new version:

- [ ] **Bump version** with `bump2version patch|minor|major`
- [ ] Update `CHANGELOG.md` with release notes
- [ ] Run full test suite: `pytest tests/`
- [ ] Build distributions: `python -m build`
- [ ] Test installation from built wheel locally
- [ ] Upload to TestPyPI and test installation
- [ ] Create GitHub release with tag `v1.3.2`
- [ ] Upload to production PyPI
- [ ] Verify installation: `pip install difflut`
- [ ] Post release announcement (if applicable)

---

## Resources

- **Installation Guide**: [`docs/INSTALLATION.md`](../INSTALLATION.md) - User installation instructions
- **Version Management**: [`.github/GITHUB_GUIDE.md`](../../.github/GITHUB_GUIDE.md) - Version bumping and release workflow
- **Container Builds**: [`apptainer/README.md`](../../apptainer/README.md) - Apptainer container usage
- **Contributing Guide**: [`docs/DEVELOPER_GUIDE/contributing.md`](contributing.md) - Development workflow
- **Python Packaging**: https://packaging.python.org/ - Official packaging documentation
- **PyPI Help**: https://pypi.org/help/ - PyPI guidelines and best practices
