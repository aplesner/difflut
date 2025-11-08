# DiffLUT Test Suite

Comprehensive test suite for the DiffLUT library.

## Overview

**Total Tests**: 641+ tests across all components

## Directory Structure

```
tests/
├── test_nodes/          # Node tests (477 tests)
├── test_layers/         # Layer tests (164 tests)
├── test_encoders/       # Encoder tests (48 tests)
├── test_utils/          # Utility tests (16 tests)
└── test_registry_validation.py  # Registry validation
```

## Test Organization

### `test_nodes/` (477 tests)
Comprehensive tests for all node types, initializers, and regularizers.

**Files**:
- `test_node_functionality.py` - Core node functionality (96 tests)
- `test_initializers.py` - Initializer validation (115 tests)
- `test_regularizers.py` - Regularizer validation (58 tests)
- `test_node_initializer_integration.py` - Node × Initializer (152 tests)
- `test_node_regularizer_integration.py` - Node × Regularizer (72 tests)
- `test_node_combined_features.py` - Combined features (17 tests)

See [test_nodes/README.md](test_nodes/README.md) for details.

### `test_layers/` (164 tests)
Tests for layer implementations.

**Files**:
- `test_layer_functionality.py` - Core layer functionality
- `test_fused_kernels.py` - Fused kernel operations
- `test_convolutional_learning.py` - Convolutional layer learning (marked `slow`)

### `test_encoders/` (48 tests)
Tests for encoder implementations.

**Files**:
- `test_encoder_functionality.py` - Core encoder functionality

### `test_utils/` (16 tests)
Tests for utility modules.

**Files**:
- `test_utils_modules.py` - Utility module tests (e.g., GroupSum)

### `test_registry_validation.py`
Registry validation tests ensuring all registered components are implemented.

## Pytest Markers

The test suite uses the following markers:

- **`gpu`**: Tests that require GPU (automatically skipped if CUDA unavailable)
- **`slow`**: Time-intensive tests (e.g., convolutional learning)
- **`skip_ci`**: Tests to skip in CI/CD pipeline
- **`experimental`**: Tests for experimental/WIP features

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run by Directory
```bash
pytest tests/test_nodes/
pytest tests/test_layers/
pytest tests/test_encoders/
pytest tests/test_utils/
```

### Run with Markers
```bash
# Only GPU tests
pytest tests/ -m "gpu"

# Skip slow tests
pytest tests/ -m "not slow"

# Skip CI-excluded tests
pytest tests/ -m "not skip_ci and not experimental"
```

### Run Specific File
```bash
pytest tests/test_nodes/test_node_functionality.py -v
```

### Collect Tests Without Running
```bash
pytest tests/ --collect-only
```

## CI/CD Integration

GitHub Actions workflow automatically runs:
```bash
pytest tests/ -v -m "not gpu and not skip_ci and not experimental"
```

This excludes:
- GPU-specific tests (CI uses CPU)
- Tests marked for CI skip
- Experimental/WIP tests

## Test Patterns

### Parametrized Tests
Most tests use pytest parametrization for comprehensive coverage:
```python
@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
def test_node_feature(node_name):
    # Test runs once for each registered node
    pass
```

### Double Parametrization (Integration Tests)
```python
@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
@pytest.mark.parametrize("init_name", REGISTRY.list_initializers())
def test_combination(node_name, init_name):
    # Test runs for every node-initializer combination
    pass
```

### Test Utilities
Common test utilities are in `testing_utils.py`:
- `generate_uniform_input()` - Generate test tensors
- `instantiate_node()` - Create node instances
- `compare_cpu_gpu_forward()` - CPU/GPU consistency checks
- `assert_gradients_exist()` - Gradient validation
- `IgnoreWarnings()` - Context manager for expected warnings

## Configuration

### `pyproject.toml`
Contains pytest configuration:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-ra -q --strict-markers"
markers = [
    "slow: time-intensive tests",
    "gpu: tests requiring GPU",
    "skip_ci: skip in CI/CD",
    "experimental: experimental features"
]
```

### Import Structure
Tests use absolute imports via `conftest.py`:
```python
# tests/conftest.py adds tests/ to sys.path
from testing_utils import ...  # Absolute import
```

## Adding New Tests

1. **Choose appropriate directory** based on component type
2. **Use clear, descriptive file names** (e.g., `test_node_functionality.py`)
3. **Add markers** for GPU/slow/experimental tests
4. **Use parametrization** for comprehensive coverage
5. **Update README** if adding new test files
6. **Run locally** before committing:
   ```bash
   pytest tests/ -v
   pytest tests/ -m "not slow"  # Quick check
   ```

## Coverage Goals

- ✅ All registered nodes tested
- ✅ All registered initializers tested
- ✅ All registered regularizers tested
- ✅ All registered layers tested
- ✅ All registered encoders tested
- ✅ CPU/GPU consistency validated
- ✅ Gradient flow validated
- ✅ Shape correctness validated
- ✅ Output range validation

## Troubleshooting

### Tests Not Discovered
- Ensure file names match `test_*.py` pattern
- Check that `__init__.py` exists in subdirectories
- Verify `conftest.py` is present for import resolution

### Import Errors
- Check that `conftest.py` adds tests directory to path
- Use absolute imports for testing_utils

### Marker Warnings
- All markers must be defined in `pyproject.toml`
- Use `pytest --markers` to see available markers
