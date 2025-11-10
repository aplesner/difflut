# DiffLUT Test Suite

This directory contains the test suite for the DiffLUT library.

## Running Tests

### Run All Tests

```bash
pytest tests/
```

Or with verbose output:

```bash
pytest tests/ -v
```

### Run a Specific Test File

```bash
pytest tests/test_grouped_convolutional_connections.py
```

### Run a Specific Test Class

```bash
pytest tests/test_utils_modules.py::TestGroupSum
```

### Run a Specific Test Function

```bash
pytest tests/test_grouped_convolutional_connections.py::test_grouped_connections_coverage
```

### Run a Specific Parametrized Test

To run a specific parameter combination:

```bash
pytest tests/test_grouped_convolutional_connections.py::test_grouped_connections_coverage[42]
```

### Run Tests by Marker

Run only GPU tests:

```bash
pytest tests/ -m gpu
```

Skip GPU tests:

```bash
pytest tests/ -m "not gpu"
```

Run slow training tests (requires GPU):

```bash
pytest tests/ -m "slow and gpu and training"
```

Skip slow tests (recommended for quick test runs):

```bash
pytest tests/ -m "not slow"
```

### Additional Options

Show print statements during test execution:

```bash
pytest tests/ -s
```

Stop at first failure:

```bash
pytest tests/ -x
```

Show local variables on failure:

```bash
pytest tests/ -l
```

Run tests in parallel (requires pytest-xdist):

```bash
pytest tests/ -n auto
```

## Test Organization

- **test_nodes_forward_pass.py**: Tests for individual node forward passes
- **test_layers_forward_pass.py**: Tests for layer forward passes
- **test_convolutional_learning.py**: Learning tests for convolutional layers
- **test_grouped_convolutional_connections.py**: Tests for grouped convolutional connections
- **test_train_grouped_connections.py**: Training tests for grouped connections (slow, GPU-only)
- **test_utils_modules.py**: Tests for utility modules like GroupSum
- **test_fused_kernels.py**: Tests for fused kernel operations
- **test_encoders_forward_pass.py**: Tests for encoder modules
- **test_registry_validation.py**: Tests for registry validation

## Writing New Tests

When writing new tests:

1. Use pytest fixtures and parametrization for comprehensive testing
2. Import test utilities from `testing_utils.py`
3. Use descriptive test function names starting with `test_`
4. Add docstrings explaining what each test validates
5. Use `@pytest.mark.gpu` for tests that require CUDA
6. Parametrize tests with multiple seeds for robustness

Example:

```python
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_my_feature(seed):
    """Test my feature across multiple seeds."""
    from testing_utils import generate_uniform_input

    input_tensor = generate_uniform_input((4, 10), seed=seed)
    # ... test implementation
```
