# DiffLUT Installation Guide

> This document is for **users** installing DiffLUT now (from source) or in the future (via PyPI), plus a concise CUDA & container overview. For packaging, version management, branching strategy, formatting, and contribution workflow please see:
> - Developer Contributing Guide: `docs/DEVELOPER_GUIDE/contributing.md`
> - GitHub Workflow Guide: `.github/GITHUB_GUIDE.md`
> - Packaging & Distribution: `docs/DEVELOPER_GUIDE/packaging.md`
> - Apptainer Container Usage: `apptainer/README.md`

---

## 1. Current Status & Availability

DiffLUT is **currently not published on PyPI**. You must install it from source. When the package is released, commands like:

```bash
pip install difflut            # (future) auto-detect GPU
pip install difflut[gpu]       # (future) force CUDA build
pip install difflut[cpu]       # (future) skip CUDA extensions
pip install difflut[dev]       # (future) include development extras
```

will become available. Until then, use the source installation instructions below.

---

## 2. Minimum Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| Python | 3.10+ | Required for modern typing (PEP 604 usage) |
| PyTorch | 1.9.0+ | CUDA optional; CPU fallback works |
| NumPy | 1.19.0+ | General tensor preprocessing |
| CUDA Toolkit | Optional | Needed only for GPU-accelerated nodes |

### GPU‑Accelerated Optional Nodes (CPU fallback exists)
`fourier_cuda`, `hybrid_cuda`, `efd_cuda` (DWN), `dwn_stable_cuda`, `probabilistic_cuda`, `learnable_mapping_cuda`, `mapping_cuda`.

---

## 3. Tested Version Matrix (Derived from CI Workflows)

The CI (`.github/workflows/tests.yml`) runs the full test suite across these combinations:

| Python | PyTorch | CUDA (GPU jobs) | CPU Tests | GPU Tests |
|--------|---------|-----------------|-----------|-----------|
| 3.10 | 2.4.0 | 12.4 | ✅ | ✅ |
| 3.10 | 2.5.0 | 12.4 | ✅ | ✅ |
| 3.11 | 2.6.0 | 12.6 | ✅ | ✅ |
| 3.11 | 2.7.0 | 12.6 | ✅ | ✅ |
| 3.12 | 2.8.0 | 12.8 | ✅ | ✅ |
| 3.12 | 2.9.0 | 12.8 | ✅ | ✅ |

Older Python versions (3.7–3.9) appear in metadata for historical compatibility, but **active testing starts at 3.10**. Use 3.10 or later for production.

---

## 4. Install From Source (Current Method)

```bash
git clone https://github.com/aplesner/difflut.git
cd difflut/difflut

# Standard install (attempts CUDA if available)
pip install .

# Force CPU-only build
pip install .[cpu]

# Development (editable + dev extras)
pip install -e .[dev]
```

You can combine extras in editable mode:

```bash
pip install -e .[dev,gpu]
pip install -e .[dev,cpu]
```

### Verifying Installation
```python
import torch, difflut
print("DiffLUT", difflut.__version__)
print("PyTorch", torch.__version__)
print("CUDA available", torch.cuda.is_available())
```

---

## 5. CUDA Build Behavior

During setup DiffLUT checks:
1. `CUDA_HOME` or `CUDA_PATH` environment variable
2. `torch.cuda.is_available()`
3. Presence of matching `.cpp` + `_kernel.cu` pairs for extensions

If any CUDA criteria fail, it silently falls back to CPU implementations.

### Selecting a Specific CUDA Toolkit
```bash
export CUDA_HOME=/usr/local/cuda-12.1    # adjust path
pip install -e .[dev,gpu]
```

### Quick CUDA Sanity
```bash
python - <<'PY'
import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
```

---

## 6. Apptainer Container (Recommended for HPC)

Use the maintained container for reproducible GPU environments. See detailed instructions in `apptainer/README.md`.

```bash
# GPU execution
apptainer exec --nv apptainer/difflut.sif python -c "import difflut; print(difflut.__version__)"

# CPU-only execution
apptainer exec apptainer/difflut.sif python -c "import difflut; print(difflut.__version__)"

# Interactive shell
apptainer shell --nv apptainer/difflut.sif
```

For remote sync & architecture flags, consult the container README.

---

## 7. Troubleshooting (Common Issues)

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| CUDA extensions not built | Missing toolkit or incompatible PyTorch wheel | Set `CUDA_HOME`; reinstall with `[gpu]` |
| ImportError for a `_cuda` module | Build failed silently | Reinstall with `pip install --no-build-isolation -e .[gpu]` to view logs |
| Slow install / wheel build errors | Build isolation / missing dev tools | `pip install --no-build-isolation .` or add `build-essential` (Linux) |
| PyTorch reports CUDA False | Using CPU wheel | Reinstall PyTorch with correct CUDA index URL |

### Helpful Commands
```bash
nvcc --version                # confirm toolkit
echo $CUDA_HOME               # path should exist
python -c "import torch;print(torch.version.cuda)"  # reported CUDA from PyTorch
python -c "import difflut,torch;print(difflut.__version__, torch.cuda.is_available())"
```

### For More Advanced Packaging / Publishing
Refer to: `docs/DEVELOPER_GUIDE/packaging.md` (wheel/sdist builds, TestPyPI, release process).

---

## 8. Quick Reference (Source Era vs Future PyPI)

| Scenario | Current (Source) | Future (PyPI) |
|----------|------------------|---------------|
| Standard install | `pip install .` | `pip install difflut` |
| Force CPU | `pip install .[cpu]` | `pip install difflut[cpu]` |
| Force GPU | `pip install .[gpu]` | `pip install difflut[gpu]` |
| Dev editable | `pip install -e .[dev]` | `pip install difflut[dev]` |

---

## 9. Developer References

- Contribution workflow, formatting (Black/isort), branching, version bumps: see `docs/DEVELOPER_GUIDE/contributing.md` and `.github/GITHUB_GUIDE.md`.
- Packaging, TestPyPI & releases: `docs/DEVELOPER_GUIDE/packaging.md`.
- Container details: `apptainer/README.md`.

---

## 10. Getting Help

| Resource | Location |
|----------|----------|
| Issues / Bug Reports | https://github.com/aplesner/difflut/issues |
| Documentation Root | `docs/` directory |
| Apptainer Guide | `apptainer/README.md` |
| Developer Workflow | `.github/GITHUB_GUIDE.md` |

If an installation problem persists, include: Python version, PyTorch version, CUDA version (if GPU), OS, and the first 30 lines of the build log when opening an issue.

---

_Last updated: CI workflow sync – Python 3.10–3.12, PyTorch 2.4.0–2.9.0, CUDA 12.4/12.6/12.8._
