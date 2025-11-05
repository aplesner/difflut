# DiffLUT Test Suite

Comprehensive test suite for the DiffLUT library with auto-discovery of components via the registry system.

## Test Coverage

### 1. Registry Validation (`test_registry_validation.py`)
- **Purpose**: Verify all registered components are actually implemented
- **Tests**:
  - All nodes can be instantiated
  - All layers can be instantiated
  - All encoders can be instantiated
  - All initializers are callable
  - All regularizers are callable
  - Registry consistency checks

### 2. Node Forward Pass (`test_nodes_forward_pass.py`)
- **Purpose**: Test each node type for correct behavior
- **Tests per node**:
  1. ✓ Correct output shape
  2. ✓ Output range [0, 1]
  3. ✓ CPU/GPU consistency (when CUDA available)
  4. ✓ Gradients exist and non-zero
  5. ✓ Works with initializers
  6. ✓ Works with regularizers

**Nodes tested**: LinearLUT, PolyLUT, NeuralLUT, DWN, DWNStable, Fourier, Hybrid, Probabilistic

### 3. Layer Forward Pass (`test_layers_forward_pass.py`)
- **Purpose**: Test each layer type with all node combinations
- **Tests per layer**:
  1. ✓ Works with all node types
  2. ✓ Correct output shape
  3. ✓ Output range [0, 1]
  4. ✓ CPU/GPU consistency (when CUDA available)
  5. ✓ Gradients exist and non-zero
  6. ✓ Different layer sizes

**Layers tested**: Random, Learnable

### 4. Encoder Forward Pass (`test_encoders_forward_pass.py`)
- **Purpose**: Test each encoder type
- **Tests per encoder**:
  1. ✓ Correct shape with flatten=True
  2. ✓ Correct shape with flatten=False
  3. ✓ Output range [0, 1] (flatten=True)
  4. ✓ Output range [0, 1] (flatten=False)
  5. ✓ Fit and encode work correctly

**Encoders tested**: Thermometer, GaussianThermometer, DistributiveThermometer, Gray, OneHot, Binary, SignMagnitude, Logarithmic

### 5. Utility Modules (`test_utils_modules.py`)
- **Purpose**: Test utility modules like GroupSum
- **Tests per module**:
  - Basic forward pass
  - Correct grouping and summing behavior
  - Tau parameter scaling
  - Batch handling with various batch sizes
  - Integration with layer outputs
  - Gradient flow through module
  - Handling of uneven grouping (padding)

**Modules tested**: GroupSum

### 6. Test Utilities (`test_utils.py`)
- **Purpose**: Shared utilities for all tests
- **Provides**:
  - Device management (CPU/GPU detection)
  - Random input generation
  - Assertion helpers
  - CPU-GPU consistency checking
  - Component discovery from registry
  - Component instantiation helpers

## Running Tests

### Run All Tests
```bash
cd tests/
python run_all_tests.py
```

### Run Individual Test Suite
```bash
cd tests/
python test_registry_validation.py
python test_nodes_forward_pass.py
python test_layers_forward_pass.py
python test_encoders_forward_pass.py
python test_utils_modules.py
```

### Run Specific Tests
Each test file can be run independently:
```bash
# Test only registry
python test_registry_validation.py

# Test only nodes
python test_nodes_forward_pass.py

# Test only layers
python test_layers_forward_pass.py

# Test only encoders
python test_encoders_forward_pass.py
```

## Auto-Discovery

Tests automatically discover components via the registry:
- **Nodes**: Uses `REGISTRY.list_nodes()`
- **Layers**: Uses `REGISTRY.list_layers()`
- **Encoders**: Uses `REGISTRY.list_encoders()`
- **Initializers**: Uses `REGISTRY.list_initializers()`
- **Regularizers**: Uses `REGISTRY.list_regularizers()`

When new components are registered, they are automatically tested!

## CUDA Testing

- GPU tests are **automatically skipped** when CUDA is not available
- CPU/GPU consistency tests only run if CUDA is available
- All tests use `is_cuda_available()` to check before GPU operations

## Test Output

Tests produce detailed output:
```
================================================================================
  NODE FORWARD PASS TESTS
================================================================================

Testing 8 nodes: ['linear_lut', 'polylut', 'neurallut', 'dwn', 'dwn_stable', 'fourier', 'hybrid', 'probabilistic']

------------------------------------------------------------------------
  Node: linear_lut
------------------------------------------------------------------------
  ✓ PASS: linear_lut: Shape
  ✓ PASS: linear_lut: Output Range [0,1]
  ✓ PASS: linear_lut: CPU/GPU Consistency
  ✓ PASS: linear_lut: Gradients
  ✓ PASS: linear_lut: With Initializer
  ✓ PASS: linear_lut: With Regularizer
  → 6 passed, 0 failed
```

## Test Statistics

| Component | Tests per | Total Tests |
|-----------|-----------|-------------|
| Nodes (8) | 6 tests | 48 tests |
| Layers (2) | 6 tests | 12 tests |
| Encoders (8) | 5 tests | 40 tests |
| Utility Modules | 7 tests | 7 tests |
| Model Training | 1 integration test | 1 test |
| Registry | 6 tests | 6 tests |
| **Total** | - | **114 tests** |

## Tolerance Settings

- **Forward pass**: atol=1e-6, rtol=1e-5
- **Gradients**: atol=1e-4, rtol=1e-3
- **CPU/GPU consistency**: atol=1e-5, rtol=1e-4

## Running Tests

### Run all tests
```bash
cd difflut/tests
python run_all_tests.py
```

### Run specific test
```bash
python test_nodes_forward_pass.py
python test_layers_forward_pass.py
python test_encoders_forward_pass.py
python test_registry_validation.py
python test_utils_modules.py
python test_model_training.py
```

## CI/CD Integration

All tests are designed to be CI/CD-friendly:

### Features
- ✓ **Deterministic**: Fixed random seeds ensure reproducible results
- ✓ **Fast**: Designed to complete in seconds to minutes
- ✓ **Clean output**: Minimal noise, warnings suppressed
- ✓ **Proper exit codes**: 0 for success, 1 for failure
- ✓ **No GPU required**: All tests run on CPU
- ✓ **No interactive output**: Suitable for automated pipelines

### Warning Suppression
All test files automatically suppress non-critical warnings:
```python
warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut')
warnings.filterwarnings('ignore', category=UserWarning, module='difflut')
```

This keeps CI/CD logs clean while preserving test output for debugging.

### Example CI/CD Usage (GitHub Actions)

```yaml
- name: Run DiffLUT Tests
  run: |
    cd difflut/tests
    python run_all_tests.py
```

### Expected Output

**Success:**
```
  TEST SUMMARY
  ✓ PASS: Test Utilities Module                              (  3.69s)
  ✓ PASS: Registry Validation Tests                          (  2.79s)
  ✓ PASS: Node Forward Pass Tests                            (  3.24s)
  ✓ PASS: Layer Forward Pass Tests                           (  2.95s)
  ✓ PASS: Encoder Forward Pass Tests                         (  2.85s)
  ✓ PASS: Utility Modules Tests (GroupSum)                   (  2.74s)
  ✓ PASS: Model Training Test                                (  1.25s)

  Total: 7/7 test suites passed (19.51s)

✓ All test suites passed!
```

**Failure:**
```
  TEST SUMMARY
  ✓ PASS: Test Utilities Module                              (  3.69s)
  ✓ PASS: Registry Validation Tests                          (  2.79s)
  ✗ FAIL: Node Forward Pass Tests                            (  3.24s)
  ...
  Total: 2/7 test suites passed (19.51s)

⚠ 5 test suite(s) failed!
```

Exit code will be 1 for failure, allowing CI/CD pipelines to stop at this point.

## Troubleshooting

### CUDA tests skipped
- This is normal if CUDA is not available
- Tests show "⊘ Skipping CUDA test (CUDA not available)"
- CPU tests still run normally

### Import errors
- Ensure you're running from the `tests/` directory
- Check that `difflut` package is installed: `pip install -e ..`

### Specific test failures
- Run the failing test individually for more details
- Check test output for the specific assertion that failed
- Use the tolerance constants in `test_utils.py` to understand acceptable ranges

## Adding New Tests

When adding new components:

1. Register them with the appropriate decorator:
   ```python
   @register_node("my_node")
   class MyNode(BaseNode):
       ...
   ```

2. Tests will automatically discover and test them
3. No changes to test files needed!

## Development

The test suite is designed to be:
- **Comprehensive**: Tests all critical properties
- **Auto-discovering**: No manual test registration needed
- **Maintainable**: Shared utilities reduce code duplication
- **Scalable**: Easy to add new component types
- **Informative**: Clear output and error messages
- **CI/CD-ready**: Deterministic, fast, clean output

## Exit Codes

- `0`: All tests passed
- `1`: One or more tests failed

