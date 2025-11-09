## Testing Strategy

DiffLUT employs a comprehensive testing framework built on `pytest` with specialized utilities for validating LUT-based components. The test suite uses automatic device detection and parameterized testing to ensure all registered components function correctly on both CPU and GPU hardware.

### Test Suite Overview

**Total Coverage**: 641+ tests across all component types

The test suite is organized into focused directories:

```
tests/
├── test_nodes/          # Node functionality and integrations (477 tests)
├── test_layers/         # Layer implementations (164 tests)
├── test_encoders/       # Input encoders (48 tests)
├── test_utils/          # Utility modules (16 tests)
├── test_registry_validation.py  # Registry consistency
├── testing_utils.py     # Shared test utilities and fixtures
└── conftest.py          # Pytest configuration and device fixture
```

**What We Test**:
- All registered nodes, layers, and encoders via parametrization
- Component initialization with various configurations
- Forward pass correctness (shape, range, gradients)
- CPU/GPU consistency for CUDA-enabled components
- Initializer and regularizer integration
- Registry completeness and component availability
- Utility module functionality

### Device Auto-Detection Strategy

Tests automatically adapt to available hardware using a `device` fixture:

**Device-Agnostic Tests** (no marker):
- Automatically use CUDA when available, CPU otherwise
- Test functionality on whatever hardware is present
- Most tests fall into this category

**GPU-Required Tests** (`@pytest.mark.gpu`):
- Explicitly require CUDA hardware
- Fail if CUDA unavailable (expected behavior)
- Used for CPU/GPU consistency checks and fused kernels

See `tests/GPU_TESTING_STRATEGY.md` for comprehensive documentation

### Pytest Markers

The test suite uses markers to categorize and filter tests:

**`gpu`**: Tests requiring CUDA hardware
- Used for CPU/GPU consistency validation
- Used for CUDA kernel correctness checks
- Fails if CUDA unavailable (intentional)
- Excluded from CI: `-m "not gpu"`

**`slow`**: Time-intensive tests
- Convolutional learning scenarios
- Large-scale integration tests
- Excluded from quick test runs: `-m "not slow"`

**`skip_ci`**: Tests excluded from CI/CD
- Tests requiring specific hardware or setup
- Manual validation tests
- Excluded from CI: `-m "not skip_ci"`

**`experimental`**: Work-in-progress features
- Unstable or incomplete functionality
- Excluded from standard runs: `-m "not experimental"`

### When to Use GPU Marker

**Use `@pytest.mark.gpu` for:**
- CPU/GPU consistency comparisons (requires both devices)
- Fused CUDA kernel validation
- CUDA extension availability checks

**Do not use for:**
- Regular node/layer functionality tests (auto-detect device)
- Shape and range validation (device-independent)
- Gradient flow tests (PyTorch handles device automatically)

Example:
```python
# Device-agnostic - automatically uses GPU when available
def test_node_forward_pass(device):
    node = DWNNode(input_dim=4, output_dim=2).to(device)
    x = torch.randn(10, 4).to(device)
    output = node(x)
    assert output.shape == (10, 2)

# GPU-required - explicitly needs CUDA
@pytest.mark.gpu
def test_cpu_gpu_consistency():
    node_cpu = DWNNode(input_dim=4, output_dim=2)
    node_gpu = DWNNode(input_dim=4, output_dim=2).cuda()
    # Compare CPU vs GPU outputs
```

### Using Testing Utilities

DiffLUT provides comprehensive testing utilities in `testing_utils.py`:

```python
from testing_utils import (
    # Device management
    get_device,              # Returns 'cuda' if available, else 'cpu'
    get_available_devices,   # List of available devices
    is_cuda_available,       # Check CUDA availability
    skip_if_no_cuda,         # Decorator for CUDA-only tests
    
    # Data generation
    generate_random_input,   # Random values
    generate_binary_input,   # Binary [0, 1] values
    generate_uniform_input,  # Uniform [0, 1] values
    
    # Assertions
    assert_shape_equal,      # Shape validation
    assert_range,            # Range validation
    assert_gradients_exist,  # Gradient flow validation
    assert_tensors_close,    # Tensor comparison
    
    # Component instantiation
    instantiate_node,        # Create node with default warnings
    instantiate_layer,       # Create layer with default warnings
    instantiate_encoder,     # Create encoder with default warnings
    
    # Context managers
    IgnoreWarnings,          # Suppress expected warnings
    
    # Tolerance constants
    FP32_ATOL,              # Forward pass absolute tolerance
    FP32_RTOL,              # Forward pass relative tolerance
    GRAD_ATOL,              # Gradient absolute tolerance
    GRAD_RTOL,              # Gradient relative tolerance
    CPU_GPU_ATOL,           # CPU/GPU absolute tolerance
    CPU_GPU_RTOL,           # CPU/GPU relative tolerance
)
```

### Writing Node Tests

Follow the pattern used in `test_nodes/test_node_functionality.py`:

```python
"""Test suite for custom node."""
import pytest
import torch
from testing_utils import (
    IgnoreWarnings,
    assert_shape_equal,
    assert_range,
    assert_gradients_exist,
    generate_uniform_input,
    instantiate_node,
    is_cuda_available,
    CPU_GPU_ATOL,
    CPU_GPU_RTOL,
)
from difflut.registry import REGISTRY


class TestMyCustomNode:
    """Comprehensive tests for MyCustomNode."""
    
    def test_shape_correct(self, device):
        """Test 1: Forward pass produces correct output shape."""
        from difflut.nodes import MyCustomNode
        
        with IgnoreWarnings():
            node = MyCustomNode(input_dim=4, output_dim=2).to(device)
        
        batch_size = 8
        input_tensor = generate_uniform_input((batch_size, 4), device=device)
        
        with torch.no_grad():
            output = node(input_tensor)
        
        expected_shape = (batch_size, 2)
        assert_shape_equal(output, expected_shape)
    
    def test_output_range(self, device):
        """Test 2: Output range is [0, 1]."""
        from difflut.nodes import MyCustomNode
        
        with IgnoreWarnings():
            node = MyCustomNode(input_dim=4, output_dim=1).to(device)
        node.eval()
        
        for seed in [42, 123, 456]:
            input_tensor = generate_uniform_input((8, 4), seed=seed, device=device)
            with torch.no_grad():
                output = node(input_tensor)
            assert_range(output, 0.0, 1.0)
    
    @pytest.mark.gpu
    def test_cpu_gpu_consistency(self):
        """Test 3: CPU and GPU produce same results."""
        if not is_cuda_available():
            pytest.fail("Test requires CUDA but it's not available")
        
        from difflut.nodes import MyCustomNode
        
        with IgnoreWarnings():
            node_cpu = MyCustomNode(input_dim=4, output_dim=1)
            node_gpu = MyCustomNode(input_dim=4, output_dim=1).cuda()
        
        node_gpu.load_state_dict(node_cpu.state_dict())
        
        input_cpu = generate_uniform_input((8, 4), seed=42, device="cpu")
        input_gpu = input_cpu.cuda()
        
        with torch.no_grad():
            output_cpu = node_cpu(input_cpu)
            output_gpu = node_gpu(input_gpu).cpu()
        
        torch.testing.assert_close(
            output_cpu, output_gpu, atol=CPU_GPU_ATOL, rtol=CPU_GPU_RTOL
        )
    
    def test_gradients_exist(self, device):
        """Test 4: Gradients flow correctly through node."""
        from difflut.nodes import MyCustomNode
        
        with IgnoreWarnings():
            node = MyCustomNode(input_dim=4, output_dim=1).to(device)
        
        node.train()
        input_tensor = generate_uniform_input((8, 4), seed=42, device=device)
        input_tensor.requires_grad = True
        
        output = node(input_tensor)
        loss = output.sum()
        loss.backward()
        
        assert_gradients_exist(node)
        assert input_tensor.grad is not None
    
    def test_regularization(self, device):
        """Test 5: Regularization computation."""
        from difflut.nodes import MyCustomNode
        from difflut.nodes.node_config import NodeConfig
        
        node_config = NodeConfig(input_dim=4, output_dim=1)
        node = MyCustomNode(**node_config.to_dict()).to(device)
        
        reg = node.regularization()
        assert reg.ndim == 0  # Scalar
        assert reg.item() >= 0  # Non-negative
    
    def test_export_bitstream(self, device):
        """Test 6: Bitstream export for FPGA deployment."""
        from difflut.nodes import MyCustomNode
        
        with IgnoreWarnings():
            node = MyCustomNode(input_dim=4, output_dim=1).to(device)
        
        bitstream = node.export_bitstream()
        
        import numpy as np
        assert isinstance(bitstream, np.ndarray)
        assert bitstream.shape == (2**4, 1)
```

**Key Patterns**:
- Use `device` fixture parameter for automatic device detection
- Use `IgnoreWarnings()` to suppress expected initialization warnings
- Use `generate_uniform_input(..., device=device)` for device-aware data
- Only mark `test_cpu_gpu_consistency` with `@pytest.mark.gpu`
- Test shape, range, gradients, and export functionality

### Writing Encoder Tests

Follow the pattern from `test_encoders/test_encoder_functionality.py`:

```python
"""Test suite for custom encoder."""
import pytest
import torch
from testing_utils import (
    IgnoreWarnings,
    assert_shape_equal,
    assert_range,
    generate_uniform_input,
    instantiate_encoder,
)


class TestMyCustomEncoder:
    """Comprehensive tests for MyCustomEncoder."""
    
    def test_shape_flatten_true(self, device):
        """Test 1: Flatten=True produces 2D output."""
        from difflut.encoder import MyCustomEncoder
        
        with IgnoreWarnings():
            encoder = MyCustomEncoder(num_bits=8, flatten=True).to(device)
        
        train_data = generate_uniform_input((100, 50), seed=42, device=device)
        encoder.fit(train_data)
        
        test_data = generate_uniform_input((10, 50), seed=123, device=device)
        with torch.no_grad():
            output = encoder.encode(test_data)
        
        expected_shape = (10, 50 * 8)
        assert_shape_equal(output, expected_shape)
    
    def test_shape_flatten_false(self, device):
        """Test 2: Flatten=False produces 3D output."""
        from difflut.encoder import MyCustomEncoder
        
        with IgnoreWarnings():
            encoder = MyCustomEncoder(num_bits=8, flatten=False).to(device)
        
        train_data = generate_uniform_input((100, 50), seed=42, device=device)
        encoder.fit(train_data)
        
        test_data = generate_uniform_input((10, 50), seed=123, device=device)
        with torch.no_grad():
            output = encoder.encode(test_data)
        
        expected_shape = (10, 50, 8)
        assert_shape_equal(output, expected_shape)
    
    def test_output_range(self, device):
        """Test 3: Output is binary [0, 1]."""
        from difflut.encoder import MyCustomEncoder
        
        with IgnoreWarnings():
            encoder = MyCustomEncoder(num_bits=8, flatten=True).to(device)
        
        train_data = generate_uniform_input((100, 50), seed=42, device=device)
        encoder.fit(train_data)
        
        for seed in [42, 123, 456]:
            test_data = generate_uniform_input((10, 50), seed=seed, device=device)
            with torch.no_grad():
                output = encoder.encode(test_data)
            assert_range(output, 0.0, 1.0)
    
    def test_fit_required(self, device):
        """Test 4: Encoding without fitting raises error."""
        from difflut.encoder import MyCustomEncoder
        
        encoder = MyCustomEncoder(num_bits=8)
        test_data = generate_uniform_input((10, 50), device=device)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            encoder.encode(test_data)
```

### Writing Layer Tests

Follow the pattern from `test_layers/test_layer_functionality.py`:

```python
"""Test suite for custom layer."""
import pytest
import torch
from testing_utils import (
    IgnoreWarnings,
    assert_shape_equal,
    assert_gradients_exist,
    generate_uniform_input,
    instantiate_layer,
)


class TestMyCustomLayer:
    """Comprehensive tests for MyCustomLayer."""
    
    def test_shape_correct(self, device):
        """Test 1: Layer produces correct output shape."""
        from difflut.layers import MyCustomLayer
        from difflut.nodes import LinearLUTNode
        from difflut.nodes.node_config import NodeConfig
        
        node_config = NodeConfig(input_dim=4, output_dim=1)
        
        with IgnoreWarnings():
            layer = MyCustomLayer(
                input_size=256,
                output_size=128,
                node_type=LinearLUTNode,
                n=4,
                node_kwargs=node_config
            ).to(device)
        
        input_tensor = generate_uniform_input((8, 256), device=device)
        
        with torch.no_grad():
            output = layer(input_tensor)
        
        expected_shape = (8, 128)
        assert_shape_equal(output, expected_shape)
    
    def test_gradients_flow(self, device):
        """Test 2: Gradients flow through layer."""
        from difflut.layers import MyCustomLayer
        from difflut.nodes import LinearLUTNode
        from difflut.nodes.node_config import NodeConfig
        
        node_config = NodeConfig(input_dim=4, output_dim=1)
        
        with IgnoreWarnings():
            layer = MyCustomLayer(
                input_size=256,
                output_size=128,
                node_type=LinearLUTNode,
                n=4,
                node_kwargs=node_config
            ).to(device)
        
        layer.train()
        input_tensor = generate_uniform_input((8, 256), seed=42, device=device)
        input_tensor.requires_grad = True
        
        output = layer(input_tensor)
        loss = output.sum()
        loss.backward()
        
        assert_gradients_exist(layer)
```

### Parameterized Testing with Registry

For comprehensive coverage across all components, use pytest parametrization with the registry:

```python
"""Test all registered components."""
import pytest
from difflut.registry import REGISTRY
from testing_utils import IgnoreWarnings, instantiate_node


@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
def test_node_instantiation(node_name, device):
    """Test that all registered nodes can be instantiated."""
    node_class = REGISTRY.get_node(node_name)
    
    with IgnoreWarnings():
        node = instantiate_node(node_class, input_dim=4, output_dim=1).to(device)
    
    assert node is not None
    assert hasattr(node, 'forward')


@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
def test_node_forward_pass(node_name, device):
    """Test forward pass for all registered nodes."""
    node_class = REGISTRY.get_node(node_name)
    
    with IgnoreWarnings():
        node = instantiate_node(node_class, input_dim=4, output_dim=1).to(device)
    
    import torch
    x = torch.rand(8, 4).to(device)
    
    with torch.no_grad():
        output = node(x)
    
    assert output.shape == (8, 1)
```

This approach ensures all registered components are automatically tested without manually updating test files.

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_nodes/test_node_functionality.py

# Run specific test class
pytest tests/test_nodes/test_node_functionality.py::TestNodeForwardPass

# Run specific test method
pytest tests/test_nodes/test_node_functionality.py::TestNodeForwardPass::test_shape_correct

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=difflut --cov-report=html

# Run only GPU tests (requires CUDA)
pytest tests/ -m gpu

# Run excluding GPU tests (CPU only)
pytest tests/ -m "not gpu"

# Run fast tests only
pytest tests/ -m "not slow and not gpu"

# Run tests matching pattern
pytest tests/ -k "custom"

# Run with output capture disabled (see print statements)
pytest tests/ -s
```

### Local Testing Strategies

**CPU-Only Development** (no GPU available):
```bash
# Run all device-agnostic tests
pytest tests/ -m "not gpu"
```

**GPU Validation** (CUDA available):
```bash
# Run all tests (device-agnostic + GPU-required)
pytest tests/

# Or use the GPU test script
./tests/test_gpu.sh --fast   # Quick validation
./tests/test_gpu.sh --all    # Full test suite
```

**Quick Validation**:
```bash
# Fast tests only (exclude slow and GPU)
pytest tests/ -m "not slow and not gpu"
```

### CI/CD Integration

**GitHub Actions Workflow** (`.github/workflows/tests.yml`):

The CI workflow runs automatically on push and pull requests with the following strategy:

```yaml
# Fast tests (CPU only)
pytest tests/ -v -m "not slow and not gpu and not skip_ci and not experimental"

# All tests with coverage (CPU only)
pytest tests/ -v -m "not gpu and not skip_ci and not experimental" --cov=difflut
```

**What CI Tests:**
- All device-agnostic tests using CPU implementations
- Component functionality across all registered nodes, layers, and encoders
- Initializer and regularizer integration
- Registry consistency and completeness
- Utility module functionality

**What CI Excludes:**
- GPU-required tests (no CUDA hardware in GitHub Actions)
- Slow tests (in fast test run only)
- Tests marked `skip_ci` or `experimental`

**GPU Testing in CI:**

GitHub Actions standard runners do not have NVIDIA GPU hardware. Tests marked with `@pytest.mark.gpu` are excluded from CI runs using the `-m "not gpu"` filter.

**Workflow Steps:**
1. GPU Testing Warning - Displays message about GPU test exclusion
2. Run Fast Tests - Quick validation excluding slow and GPU tests
3. Run All Tests - Comprehensive coverage excluding GPU tests
4. PR Comment - Notifies developers about local GPU testing requirement

**Developer Responsibilities:**

Before merging code that affects GPU functionality:
1. Run GPU tests locally: `./tests/test_gpu.sh --fast`
2. Verify CPU/GPU consistency tests pass
3. Ensure CUDA kernels function correctly

**Self-Hosted GPU Runners:**

Work is in progress to add self-hosted GPU runners for automated GPU testing in CI. Until then, GPU validation must be performed locally on machines with NVIDIA hardware.

### Test Best Practices

1. **Use device fixture** - Accept `device` parameter for automatic device detection
2. **Use IgnoreWarnings()** - Suppress expected warnings during component instantiation
3. **Use testing_utils functions** - Ensure consistency across test suite
4. **Test multiple random seeds** - Verify robustness across different inputs
5. **Mark GPU tests appropriately** - Only use `@pytest.mark.gpu` for tests requiring CUDA
6. **Provide descriptive names** - Follow pattern `test_<description>`
7. **Include docstrings** - Explain what each test validates
8. **Test edge cases** - Empty batches, extreme values, boundary conditions
9. **Check shapes explicitly** - Use `assert_shape_equal()` for clarity
10. **Verify gradients exist** - Use `assert_gradients_exist()` helper
11. **Test deterministically** - Use fixed seeds for reproducibility
12. **Ensure graceful degradation** - GPU tests should fail cleanly without CUDA

### Adding Tests for New Components

When contributing new components:

1. **Create test file** following naming convention: `test_<component>_<category>.py`
2. **Use parametrization** if testing across multiple component types
3. **Include basic tests**: shape, range, gradients, device consistency
4. **Add component-specific tests** for unique functionality
5. **Update test documentation** if introducing new patterns
6. **Run locally before submitting**: `pytest tests/ -v`
7. **Verify GPU tests** if component has CUDA support: `./tests/test_gpu.sh --fast`

For detailed examples and patterns, see existing test files in `tests/test_nodes/`, `tests/test_layers/`, and `tests/test_encoders/`.
