# Node Tests

This directory contains comprehensive tests for all node types.

## Test Files

### `test_node_functionality.py`
- **Purpose**: Core functionality tests for all node types
- **Test Count**: ~96 tests (12 tests × 8 nodes)
- **Coverage**:
  - Shape correctness
  - Output range validation [0,1]
  - CPU/GPU consistency
  - Gradient computation
  - Different batch sizes
  - Different dimensions

### `test_initializers.py`
- **Purpose**: Test all registered initializers
- **Test Count**: ~115 tests
- **Coverage**:
  - Callable verification
  - Signature validation
  - Tensor application
  - Node parameter initialization
  - Deterministic behavior (with seed)
  - Value range validation

### `test_regularizers.py`
- **Purpose**: Test all registered regularizers
- **Test Count**: ~58 tests
- **Coverage**:
  - Callable verification
  - Signature validation
  - Node application
  - Different node type compatibility
  - Differentiability
  - Parameter sensitivity

### `test_node_initializer_integration.py`
- **Purpose**: Integration tests for nodes with initializers
- **Test Count**: ~152 tests (8 nodes × 19 initializers)
- **Coverage**:
  - All node-initializer combinations
  - Forward pass validity
  - Parameter initialization correctness
  - NaN/Inf detection

### `test_node_regularizer_integration.py`
- **Purpose**: Integration tests for nodes with regularizers
- **Test Count**: ~72 tests (8 nodes × 9 regularizers)
- **Coverage**:
  - All node-regularizer combinations
  - Regularizer computation
  - Compatibility validation
  - NaN/Inf detection

### `test_node_combined_features.py`
- **Purpose**: Integration tests for nodes with combined features
- **Test Count**: ~17 tests
- **Coverage**:
  - Nodes with both initializer and regularizer (8 tests)
  - Nodes with multiple regularizers (8 tests, marked as slow)
  - Integration summary (1 test)

## Running Tests

```bash
# Run all node tests
pytest tests/test_nodes/

# Run specific test file
pytest tests/test_nodes/test_node_functionality.py

# Run integration tests only
pytest tests/test_nodes/test_node_*_integration.py

# Run with specific markers
pytest tests/test_nodes/ -m "gpu"
pytest tests/test_nodes/ -m "slow"
pytest tests/test_nodes/ -m "not slow"
```

## Registered Components (Auto-discovered)

- **Nodes**: 8 types (dwn, dwn_stable, fourier, hybrid, linear_lut, neurallut, polylut, probabilistic)
- **Initializers**: 19 functions (glorot, he, kaiming, xavier, normal, uniform, zeros, ones, etc.)
- **Regularizers**: 9 functions (l1, l2, fourier, walsh, spectral, functional, etc.)

## Test Organization Philosophy

Tests are organized by concern:
- **Functionality tests**: Test individual components in isolation
- **Integration tests**: Test combinations of components working together
- **Combined feature tests**: Test complex scenarios with multiple features

This organization makes it easy to:
- Quickly identify which component has issues
- Run targeted test subsets
- Maintain and extend test coverage
