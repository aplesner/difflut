# Node Testing Strategy

This document describes the comprehensive testing strategy for DiffLUT nodes, initializers, and regularizers.

## Test File Organization

### 1. `test_nodes_forward_pass.py`
**Purpose**: Core node functionality tests
**Markers**: `gpu` for GPU-specific tests
**Tests**:
- Shape correctness
- Output range validation [0, 1]
- CPU/GPU consistency
- Gradient computation
- Different batch sizes
- Different dimensions

### 2. `test_initializers.py`
**Purpose**: Test all registered initializers independently
**Markers**: None (all CPU tests)
**Tests**:
- Initializer is callable
- Initializer has valid signature
- Can be applied to tensors
- Works with node parameters
- Produces deterministic results
- Produces values in reasonable range
- Registry has initializers registered

**Example initializers tested**:
- Uniform initialization
- Xavier/Glorot initialization
- Kaiming/He initialization
- Constant initialization
- etc.

### 3. `test_regularizers.py`
**Purpose**: Test all registered regularizers independently
**Markers**: None (all CPU tests)
**Tests**:
- Regularizer is callable
- Regularizer has valid signature
- Can be applied to simple nodes
- Works with different node types
- Is differentiable (produces gradients)
- Scales with parameter changes
- Registry has regularizers registered

**Example regularizers tested**:
- L1/L2 functional regularization
- Spectral/Fourier regularization
- Custom regularizers

### 4. `test_nodes_with_init_reg.py`
**Purpose**: Integration tests for nodes with initializers and regularizers
**Markers**: `slow` for comprehensive combination tests
**Tests**:
- Each node × each initializer (N × M combinations)
- Each node × each regularizer (N × K combinations)
- Nodes with both initializer and regularizer
- Nodes with multiple regularizers simultaneously

## Test Execution Examples

### Run all node tests
```bash
pytest tests/test_nodes/ -v
```

### Run only fast tests (exclude slow integration tests)
```bash
pytest tests/test_nodes/ -v -m "not slow"
```

### Run only GPU tests
```bash
pytest tests/test_nodes/ -v -m "gpu"
```

### Run specific test file
```bash
pytest tests/test_nodes/test_initializers.py -v
pytest tests/test_nodes/test_regularizers.py -v
pytest tests/test_nodes/test_nodes_with_init_reg.py -v
```

### Run integration tests (slow)
```bash
pytest tests/test_nodes/test_nodes_with_init_reg.py -v -m "slow"
```

### Run specific combination
```bash
pytest tests/test_nodes/test_nodes_with_init_reg.py::test_node_with_initializer[dwn-uniform] -v
pytest tests/test_nodes/test_nodes_with_init_reg.py::test_node_with_regularizer[linear_lut-l2] -v
```

## Test Coverage Matrix

Given:
- **N nodes** (e.g., 8: dwn, dwn_stable, fourier, hybrid, linear_lut, neurallut, polylut, probabilistic)
- **M initializers** (e.g., 5+: uniform, xavier, kaiming, etc.)
- **K regularizers** (e.g., 3+: l1, l2, spectral, etc.)

### Coverage:
| Test Type | Coverage | Test Count |
|-----------|----------|------------|
| Basic Node Tests | N nodes × 4 core tests | ~32 |
| Node × Batch Size | N nodes × 4 sizes | ~32 |
| Node × Dimensions | N nodes × 4 configs | ~32 |
| Initializer Tests | M initializers × 7 tests | ~35+ |
| Regularizer Tests | K regularizers × 6 tests | ~18+ |
| Node × Initializer | N × M combinations | ~40+ |
| Node × Regularizer | N × K combinations | ~24+ |
| **Total** | | **~213+ tests** |

## Benefits of This Structure

### 1. **Modularity**
- Each component (nodes, initializers, regularizers) tested independently
- Easy to identify which component is failing

### 2. **Comprehensive Coverage**
- All registered initializers and regularizers automatically tested
- No manual updates needed when new components are registered
- Tests all combinations systematically

### 3. **Performance**
- Fast tests (basic functionality) run quickly
- Slow tests (combinations) marked with `slow` marker
- CI can run fast tests, local dev can run all

### 4. **Debugging**
- Failed tests clearly indicate:
  - Which node failed
  - Which initializer/regularizer failed
  - Which combination failed
- Parametrized tests show exact failing combination

### 5. **Documentation**
- Tests serve as usage examples
- Shows correct API usage for init_fn and regularizers
- Demonstrates expected behavior

## Adding New Components

### Adding a New Node
1. Register with `@register_node("new_node")`
2. All tests automatically include it
3. Run `pytest tests/test_nodes/ -v -k "new_node"` to verify

### Adding a New Initializer
1. Register with `@register_initializer("new_init")`
2. All tests automatically include it
3. Run `pytest tests/test_nodes/test_initializers.py -v -k "new_init"` to verify

### Adding a New Regularizer
1. Register with `@register_regularizer("new_reg")`
2. All tests automatically include it
3. Run `pytest tests/test_nodes/test_regularizers.py -v -k "new_reg"` to verify

## CI/CD Integration

### Fast CI Run (every commit)
```bash
pytest tests/test_nodes/ -v -m "not slow and not gpu"
```

### Full CI Run (nightly/pre-merge)
```bash
pytest tests/test_nodes/ -v -m "not gpu"  # All CPU tests including slow
```

### GPU CI Run
```bash
pytest tests/test_nodes/ -v -m "gpu"
```

## Maintenance

### Regular Checks
- Verify all components are registered: `pytest tests/test_nodes/test_initializers.py::test_all_initializers_registered -v`
- Verify all regularizers are registered: `pytest tests/test_nodes/test_regularizers.py::test_all_regularizers_registered -v`
- Run integration summary: `pytest tests/test_nodes/test_nodes_with_init_reg.py::test_integration_summary -v -s`

### When Tests Fail
1. Check basic node test first
2. Then check initializer/regularizer test
3. Finally check integration test
4. This narrows down the root cause quickly
