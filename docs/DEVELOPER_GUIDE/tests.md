## Testing Your Component

DiffLUT uses a comprehensive testing framework with `pytest` and specialized testing utilities. All custom components should follow these testing patterns.

### Test Structure Overview

The test suite is organized as follows:
- **`tests/testing_utils.py`**: Shared utilities, fixtures, and helper functions
- **`tests/test_*_forward_pass.py`**: Parameterized tests for all registered components
- **`tests/test_registry_validation.py`**: Registry consistency checks
- **`tests/conftest.py`**: Pytest configuration

### Using Testing Utilities

DiffLUT provides comprehensive testing utilities in `testing_utils.py`:

```python
from testing_utils import (
    # Device management
    get_available_devices,
    is_cuda_available,
    skip_if_no_cuda,
    
    # Data generation
    generate_random_input,
    generate_binary_input,
    generate_uniform_input,
    
    # Assertions
    assert_shape_equal,
    assert_range,
    assert_gradients_exist,
    assert_tensors_close,
    
    # CPU-GPU comparison
    compare_cpu_gpu_forward,
    
    # Component instantiation
    instantiate_node,
    instantiate_layer,
    instantiate_encoder,
    
    # Warning suppression
    IgnoreWarnings,
    
    # Tolerance constants
    FP32_ATOL,
    FP32_RTOL,
    GRAD_ATOL,
    GRAD_RTOL,
    CPU_GPU_ATOL,
    CPU_GPU_RTOL,
)
```

### Writing Node Tests

Follow the pattern used in `test_nodes_forward_pass.py`:

```python
"""Test suite for custom node."""
import pytest
import torch
from testing_utils import (
    IgnoreWarnings,
    assert_shape_equal,
    assert_range,
    assert_gradients_exist,
    compare_cpu_gpu_forward,
    generate_uniform_input,
    instantiate_node,
    is_cuda_available,
    CPU_GPU_ATOL,
    CPU_GPU_RTOL,
)
from difflut.registry import REGISTRY


class TestMyCustomNode:
    """Comprehensive tests for MyCustomNode."""
    
    def test_shape_correct(self):
        """Test 1: Forward pass produces correct output shape."""
        from difflut.nodes import MyCustomNode
        
        with IgnoreWarnings():
            node = MyCustomNode(input_dim=4, output_dim=2)
        
        # Input: (batch_size, input_dim)
        batch_size = 8
        input_tensor = generate_uniform_input((batch_size, 4))
        
        with torch.no_grad():
            output = node(input_tensor)
        
        # Output: (batch_size, output_dim)
        expected_shape = (batch_size, 2)
        assert_shape_equal(output, expected_shape)
    
    def test_output_range(self):
        """Test 2: Output range is reasonable (e.g., [0, 1])."""
        from difflut.nodes import MyCustomNode
        
        with IgnoreWarnings():
            node = MyCustomNode(input_dim=4, output_dim=1)
        node.eval()
        
        # Test multiple random inputs
        for seed in [42, 123, 456]:
            input_tensor = generate_uniform_input((8, 4), seed=seed)
            with torch.no_grad():
                output = node(input_tensor)
            
            # Adjust range based on your node's expected output
            assert_range(output, 0.0, 1.0)
    
    @pytest.mark.gpu
    def test_cpu_gpu_consistency(self):
        """Test 3: CPU and GPU produce same results."""
        if not is_cuda_available():
            pytest.skip("CUDA not available")
        
        from difflut.nodes import MyCustomNode
        
        with IgnoreWarnings():
            node = MyCustomNode(input_dim=4, output_dim=1)
        
        input_tensor = generate_uniform_input((8, 4), seed=42)
        
        try:
            compare_cpu_gpu_forward(
                node, input_tensor, 
                atol=CPU_GPU_ATOL, 
                rtol=CPU_GPU_RTOL
            )
        except RuntimeError as e:
            if "CUDA" in str(e):
                pytest.skip(f"CUDA error: {e}")
            raise
    
    def test_gradients_exist(self):
        """Test 4: Gradients flow correctly through node."""
        from difflut.nodes import MyCustomNode
        
        with IgnoreWarnings():
            node = MyCustomNode(input_dim=4, output_dim=1)
        
        node.train()
        input_tensor = generate_uniform_input((8, 4), seed=42)
        input_tensor.requires_grad = True
        
        # Forward pass
        output = node(input_tensor)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are non-zero
        assert_gradients_exist(node)
        assert input_tensor.grad is not None
    
    def test_regularization(self):
        """Test 5: Regularization computation (if applicable)."""
        from difflut.nodes import MyCustomNode
        from difflut.nodes.node_config import NodeConfig
        
        node_config = NodeConfig(input_dim=4, output_dim=1)
        node = MyCustomNode(**node_config.to_dict())
        
        # Check regularization returns a scalar
        reg = node.regularization()
        assert reg.ndim == 0  # Scalar
        assert reg.item() >= 0  # Non-negative
    
    def test_export_bitstream(self):
        """Test 6: Bitstream export for FPGA deployment."""
        from difflut.nodes import MyCustomNode
        
        with IgnoreWarnings():
            node = MyCustomNode(input_dim=4, output_dim=1)
        
        bitstream = node.export_bitstream()
        
        # Check output is numpy array with correct shape
        import numpy as np
        assert isinstance(bitstream, np.ndarray)
        assert bitstream.shape == (2**4, 1)  # (2^input_dim, output_dim)
```

### Writing Encoder Tests

Follow the pattern from `test_encoders_forward_pass.py`:

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
    
    def test_shape_flatten_true(self):
        """Test 1: Flatten=True produces 2D output."""
        from difflut.encoder import MyCustomEncoder
        
        with IgnoreWarnings():
            encoder = MyCustomEncoder(num_bits=8, flatten=True)
        
        # Fit on training data
        train_data = generate_uniform_input((100, 50), seed=42)
        encoder.fit(train_data)
        
        # Encode test data
        test_data = generate_uniform_input((10, 50), seed=123)
        with torch.no_grad():
            output = encoder.encode(test_data)
        
        # Output: (batch, features * num_bits)
        expected_shape = (10, 50 * 8)
        assert_shape_equal(output, expected_shape)
    
    def test_shape_flatten_false(self):
        """Test 2: Flatten=False produces 3D output."""
        from difflut.encoder import MyCustomEncoder
        
        with IgnoreWarnings():
            encoder = MyCustomEncoder(num_bits=8, flatten=False)
        
        train_data = generate_uniform_input((100, 50), seed=42)
        encoder.fit(train_data)
        
        test_data = generate_uniform_input((10, 50), seed=123)
        with torch.no_grad():
            output = encoder.encode(test_data)
        
        # Output: (batch, features, num_bits)
        expected_shape = (10, 50, 8)
        assert_shape_equal(output, expected_shape)
    
    def test_output_range(self):
        """Test 3: Output is binary [0, 1]."""
        from difflut.encoder import MyCustomEncoder
        
        with IgnoreWarnings():
            encoder = MyCustomEncoder(num_bits=8, flatten=True)
        
        train_data = generate_uniform_input((100, 50), seed=42)
        encoder.fit(train_data)
        
        for seed in [42, 123, 456]:
            test_data = generate_uniform_input((10, 50), seed=seed)
            with torch.no_grad():
                output = encoder.encode(test_data)
            
            assert_range(output, 0.0, 1.0)
    
    def test_fit_required(self):
        """Test 4: Encoding without fitting raises error."""
        from difflut.encoder import MyCustomEncoder
        
        encoder = MyCustomEncoder(num_bits=8)
        test_data = generate_uniform_input((10, 50))
        
        with pytest.raises(RuntimeError, match="not fitted"):
            encoder.encode(test_data)
```

### Writing Layer Tests

Follow the pattern from `test_layers_forward_pass.py`:

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
    
    def test_shape_correct(self):
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
            )
        
        input_tensor = generate_uniform_input((8, 256))
        
        with torch.no_grad():
            output = layer(input_tensor)
        
        # Output: (batch, output_size)
        expected_shape = (8, 128)
        assert_shape_equal(output, expected_shape)
    
    def test_gradients_flow(self):
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
            )
        
        layer.train()
        input_tensor = generate_uniform_input((8, 256), seed=42)
        input_tensor.requires_grad = True
        
        output = layer(input_tensor)
        loss = output.sum()
        loss.backward()
        
        assert_gradients_exist(layer)
```

### Parameterized Testing with Registry

For comprehensive testing across all components, use pytest parametrization:

```python
"""Test all registered custom nodes."""
import pytest
from difflut.registry import REGISTRY
from testing_utils import IgnoreWarnings, instantiate_node


@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
def test_node_instantiation(node_name):
    """Test that all registered nodes can be instantiated."""
    node_class = REGISTRY.get_node(node_name)
    
    with IgnoreWarnings():
        node = instantiate_node(node_class, input_dim=4, output_dim=1)
    
    assert node is not None
    assert hasattr(node, 'forward')


@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
def test_node_forward_pass(node_name):
    """Test forward pass for all registered nodes."""
    node_class = REGISTRY.get_node(node_name)
    
    with IgnoreWarnings():
        node = instantiate_node(node_class, input_dim=4, output_dim=1)
    
    import torch
    x = torch.rand(8, 4)
    
    with torch.no_grad():
        output = node(x)
    
    assert output.shape == (8, 1)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_my_component.py

# Run specific test class
pytest tests/test_my_component.py::TestMyCustomNode

# Run specific test method
pytest tests/test_my_component.py::TestMyCustomNode::test_shape_correct

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=difflut --cov-report=html

# Run only GPU tests (if CUDA available)
pytest tests/ -m gpu

# Run tests matching pattern
pytest tests/ -k "custom"

# Run tests with output capture disabled (see print statements)
pytest tests/ -s
```

### Test Best Practices

1. **Use `IgnoreWarnings()` context manager** to suppress expected warnings during instantiation
2. **Use `testing_utils` functions** for consistency across test suite
3. **Test multiple random seeds** to ensure robustness
4. **Mark GPU tests** with `@pytest.mark.gpu` decorator
5. **Provide descriptive test names** following pattern: `test_<number>_<description>`
6. **Include docstrings** explaining what each test validates
7. **Test edge cases**: empty batches, extreme values, boundary conditions
8. **Check shapes explicitly** using `assert_shape_equal()` rather than direct comparison
9. **Verify gradients exist** using `assert_gradients_exist()` helper
10. **Test CPU-GPU consistency** for CUDA-enabled components

### Continuous Integration

Tests run automatically on push/PR via GitHub Actions (`.github/workflows/tests.yml`). Ensure:
- All tests pass locally before pushing
- New components are covered by tests
- Tests are deterministic (use fixed seeds)
- GPU tests gracefully skip when CUDA unavailable
