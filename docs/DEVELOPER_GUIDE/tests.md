# Testing Guide

Comprehensive guide to DiffLUT's testing framework, markers, CI/CD integration, and how to add tests for custom components.

## Table of Contents

1. [Current Test Structure](#current-test-structure)
2. [Running Tests Locally](#running-tests-locally)
3. [Test Markers](#test-markers)
4. [CI/CD Workflow](#cicd-workflow)
5. [Writing Tests for Custom Components](#writing-tests-for-custom-components)
6. [Testing Utilities](#testing-utilities)
7. [GPU Testing](#gpu-testing)
8. [Common Patterns](#common-patterns)

---

## Current Test Structure

### Directory Organization

DiffLUT uses a modular test structure mirroring the main package organization:

```
tests/
├── __init__.py                          # Test package marker
├── conftest.py                          # pytest configuration & fixtures
├── testing_utils.py                     # Shared utility functions
├── test_registry_validation.py          # Registry validation tests
│
├── test_encoders/                       # Encoder tests
│   ├── __init__.py
│   └── test_encoder_functionality.py   # Encoder implementations
│
├── test_layers/                         # Layer tests
│   ├── __init__.py
│   ├── test_layer_functionality.py     # Layer implementations
│   ├── test_convolutional_learning.py  # Specialized layer tests
│   └── test_fused_kernels.py           # CUDA kernel tests
│
├── test_nodes/                          # Node tests
│   ├── __init__.py
│   ├── test_node_functionality.py      # Node implementations
│   ├── test_node_initializer_integration.py  # Initializer integration
│   └── test_node_regularizer_integration.py  # Regularizer integration
│
├── test_utils/                          # Utility module tests
│   ├── __init__.py
│   └── test_utils_modules.py           # Utility implementations
│
├── test_experimental/                   # Experimental features
│   └── __init__.py                     # (For testing new features)
│
└── test_models/                         # Model-level tests
    └── __init__.py                     # (For integration tests)
```

### Test Categories by Component

| Component | Test Location | Focus |
|-----------|--------------|-------|
| **Encoders** | `test_encoders/` | Input discretization, fitting, encoding consistency |
| **Layers** | `test_layers/` | Connectivity patterns, forward/backward pass, augmentation |
| **Nodes** | `test_nodes/` | LUT computation, initialization, regularization |
| **Utils** | `test_utils/` | Helper functions, module utilities |

---

## Running Tests Locally

### Basic Test Execution

Run all CPU tests (excluding GPU tests):

```bash
# Fast tests only (excludes slow and GPU tests)
pytest tests/ -v -m "not slow and not gpu"

# All CPU tests with coverage
pytest tests/ -v -m "not gpu"  --cov=difflut --cov-report=html

# Fast tests with short traceback
pytest tests/ -v -m "not slow and not gpu" --tb=short
```

### Running Specific Test Categories

```bash
# Test only encoders
pytest tests/test_encoders/ -v

# Test only layers
pytest tests/test_layers/ -v

# Test only nodes
pytest tests/test_nodes/ -v

# Test only a specific file
pytest tests/test_nodes/test_node_functionality.py -v

# Test only a specific test class
pytest tests/test_nodes/test_node_functionality.py::TestLinearLUTNode -v

# Test only a specific test method
pytest tests/test_nodes/test_node_functionality.py::TestLinearLUTNode::test_output_shape -v
```

### Using Test Markers

Run tests with specific markers:

```bash
# Only fast tests (exclude slow)
pytest tests/ -v -m "not slow"

# Only slow tests
pytest tests/ -v -m "slow"

# GPU tests only (requires CUDA)
pytest tests/ -v -m "gpu"

# CPU tests only (exclude GPU)
pytest tests/ -v -m "not gpu"

# Experimental tests only
pytest tests/ -v -m "experimental"

# Exclude experimental and skip_ci tests
pytest tests/ -v -m "not experimental and not skip_ci"
```

### Advanced Options

```bash
# Show print statements
pytest tests/ -v -s

# Stop on first failure
pytest tests/ -v -x

# Run only failed tests (from last run)
pytest tests/ -v --lf

# Run failed tests first, then others
pytest tests/ -v --ff

# Show top 10 slowest tests
pytest tests/ -v --durations=10

# Parallel execution (install pytest-xdist first)
pytest tests/ -v -n auto

# Generate HTML coverage report
pytest tests/ -v --cov=difflut --cov-report=html
# Open htmlcov/index.html in browser
```

---

## Test Markers

DiffLUT uses pytest markers to categorize and control test execution. Markers enable fine-grained control over which tests run in different contexts (CI, local development, GPU machines, etc.).

### Standard Markers

#### 1. `slow`

Marks tests that take significant time to run (typically > 5 seconds).

```python
import pytest

@pytest.mark.slow
def test_large_layer_convergence():
    """Test layer convergence on large dataset."""
    layer = RandomLayer(input_size=10000, output_size=1000, ...)
    # ... test code ...
```

**Usage:**
```bash
# Skip slow tests (faster development iteration)
pytest tests/ -v -m "not slow"

# Run only slow tests
pytest tests/ -v -m "slow"
```

#### 2. `gpu`

Marks tests requiring NVIDIA GPU and CUDA.

```python
import pytest

@pytest.mark.gpu
def test_dwn_cuda_kernel():
    """Test CUDA kernel for DWN node."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    node = DWNStableNode(..., extra_params={'use_cuda': True})
    x = torch.randn(32, 6).cuda()
    # ... test code ...
```

**Important Notes:**
- ⚠️ **CI/CD runs CPU tests only**: GitHub Actions runners don't have GPU hardware
- ✅ **Run locally on GPU machine**: Use full GPU test suite locally
- ✅ **GPU setup validation**: CI validates GPU setup works without actual GPU tests

**Usage:**
```bash
# Skip GPU tests (for CPU-only development)
pytest tests/ -v -m "not gpu"

# Run only GPU tests (on GPU machine)
pytest tests/ -v -m "gpu"

# Run GPU tests with short traceback
pytest tests/ -v -m "gpu" --tb=short
```

#### 3. `experimental`

Marks tests for new, in-development features. Excluded from CI runs.

```python
import pytest

@pytest.mark.experimental
def test_new_feature_prototype():
    """Test new experimental feature (not yet stable)."""
    feature = NewExperimentalFeature()
    # ... test code ...
```

**Usage:**
- Automatically excluded from CI: CI runs with `-m "not experimental"`
- Use for features under development
- Promote to stable tests when feature is production-ready

```bash
# Test new features locally
pytest tests/test_experimental/ -v -m "experimental"

# Exclude experimental from regular test runs
pytest tests/ -v -m "not experimental"
```

#### 4. `skip_ci`

Marks tests that should not run in CI but are safe to run locally.

```python
import pytest

@pytest.mark.skip_ci
def test_with_external_dependency():
    """Test requiring optional external tool."""
    # ... test code ...
```

**Usage:**
```bash
# Run locally with all tests
pytest tests/ -v

# CI automatically excludes with: -m "not skip_ci"
# Local development can run full suite: pytest tests/ -v
```

#### 5. `training`

Marks tests that involve model training or learning. Useful for identifying tests that require longer execution time and consume more resources.

```python
import pytest

@pytest.mark.training
def test_model_convergence():
    """Test that model converges on training data."""
    model = create_model()
    # ... training loop ...
    assert final_loss < initial_loss

@pytest.mark.training
@pytest.mark.slow
def test_extended_training():
    """Test extended training with many epochs."""
    model = create_model()
    # ... long training loop ...
    pass
```

**Usage:**
```bash
# Run only training tests
pytest tests/ -v -m "training"

# Skip training tests (faster development)
pytest tests/ -v -m "not training"

# Run training tests that are not slow
pytest tests/ -v -m "training and not slow"

# Run slow training tests (e.g., convergence validation)
pytest tests/ -v -m "slow and training"
```

**Common combinations:**
```bash
# Training tests on GPU
pytest tests/ -v -m "training and gpu"

# Fast training tests (mini-batch convergence checks)
pytest tests/ -v -m "training and not slow"

# All training and learning validation tests
pytest tests/ -v -m "training"
```

### Combining Markers

```python
import pytest

# Test that is both slow AND requires GPU
@pytest.mark.slow
@pytest.mark.gpu
def test_large_model_gpu():
    """Test large model training on GPU."""
    pass

# Test marked as experimental AND slow
@pytest.mark.experimental
@pytest.mark.slow
def test_experimental_slow_feature():
    """New feature that runs slowly."""
    pass

# Test marked as training AND slow (extended training)
@pytest.mark.training
@pytest.mark.slow
def test_extended_training_convergence():
    """Test convergence over many training epochs."""
    pass

# Test marked as training AND gpu (GPU training)
@pytest.mark.training
@pytest.mark.gpu
def test_gpu_training_optimization():
    """Test GPU-accelerated training."""
    pass

# Quick training test (not slow)
@pytest.mark.training
def test_quick_convergence():
    """Test quick convergence on mini-batch."""
    pass
```

Run combined markers:

```bash
# Run fast GPU tests (skip slow GPU tests)
pytest tests/ -v -m "gpu and not slow"

# Run slow tests that are NOT experimental
pytest tests/ -v -m "slow and not experimental"

# Run experimental GPU tests
pytest tests/ -v -m "gpu and experimental"

# Run all training tests
pytest tests/ -v -m "training"

# Run fast training tests (skip slow training tests)
pytest tests/ -v -m "training and not slow"

# Run GPU training tests
pytest tests/ -v -m "training and gpu"

# Run slow GPU training tests
pytest tests/ -v -m "training and gpu and slow"

# Run everything EXCEPT training tests (quick development iteration)
pytest tests/ -v -m "not training"
```

### Pytest Configuration

Markers are defined in `pyproject.toml`:

```toml
[tool:pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests requiring GPU/CUDA (deselect with '-m "not gpu"')
    experimental: marks tests for experimental features
    skip_ci: marks tests to skip in CI
```

---

## CI/CD Workflow
### CPU Test Job

```yaml
test:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      - python: 3.10, torch: 2.4.0
      - python: 3.10, torch: 2.5.0
      - python: 3.11, torch: 2.6.0
      - python: 3.11, torch: 2.7.0
      - python: 3.12, torch: 2.8.0
      - python: 3.12, torch: 2.9.0
```

**Run command:**
```bash
# First run fast test
pytest tests/ -v -m "not slow and not gpu and not skip_ci and not experimental" --tb=short
# Then the slow ones
pytest tests/ -v -m "slow and not gpu and not skip_ci and not experimental" --tb=short
```

**Markers applied:**
- ✅ `not gpu` - Skip GPU-specific tests (no GPU available)
- ✅ `not skip_ci` - Skip CI-excluded tests
- ✅ `not experimental` - Skip experimental features

### GPU Test Job

```yaml
test-gpu:
  runs-on: ubuntu-latest
  if: github.event_name == 'pull_request' || github.ref == 'refs/heads/main'
  strategy:
    matrix:
      - python: 3.10, cuda: 12.4, torch: 2.4.0
      - python: 3.10, cuda: 12.4, torch: 2.5.0
      - python: 3.11, cuda: 12.6, torch: 2.6.0
      - python: 3.11, cuda: 12.6, torch: 2.7.0
      - python: 3.12, cuda: 12.8, torch: 2.8.0
      - python: 3.12, cuda: 12.8, torch: 2.9.0
```

**Run command:**
```bash
# First run fast tests
pytest tests/ -v -m "not slow and not skip_ci and not experimental" --tb=short
# Then the slow ones
pytest tests/ -v -m "slow and not skip_ci and not experimental" --tb=short
```

#### ⚠️ Important: GPU Testing in CI

**GitHub Actions runners do NOT have NVIDIA GPU hardware.**

What happens:

| Step | Result |
|------|--------|
| CUDA Toolkit installed | ✅ Yes (apt-get) |
| PyTorch GPU wheels | ✅ Yes (downloaded) |
| DiffLUT GPU build | ✅ Yes (setup works) |
| Actual GPU tests run | ❌ No (no GPU) |
| GPU tests skip silently | ✅ Yes (expected) |

**GPU setup validation:**
- ✅ Verifies GPU installation doesn't break the build
- ✅ Tests CUDA toolkit compatibility
- ✅ Validates GPU dependencies install correctly
- ❌ Does NOT run actual GPU computation tests

**Actual GPU testing must be done locally:**

```bash
# On a machine with NVIDIA GPU:
pytest tests/ -v --tb=short
```

### Coverage Reports

CPU test job generates coverage reports:

```bash
pytest tests/ -v --cov=difflut --cov-report=xml --cov-report=html
```

Reports uploaded to Codecov:

```yaml
- name: Upload coverage reports
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
    flags: cpu-py${{ matrix.python-version }}
```

View local coverage:

```bash
# Generate HTML report
pytest tests/ --cov=difflut --cov-report=html

# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### PR Comments

When GPU tests run in CI, a comment is added to pull requests:

```
⚠️ **GPU Testing Not Available in CI**

GitHub Actions runners do **not** have NVIDIA GPU hardware.
GPU setup validation runs in CI, but actual GPU tests **must be run locally**.

### To run GPU tests locally:
cd difflut
python -m pytest tests/ -v -m gpu


**Status:** Self-hosted GPU runners coming in the future. For now, please validate GPU functionality locally before merging.

**Current CI coverage:** CPU implementations + GPU setup validation
```

---

## Writing Tests for Custom Components

### Test File Organization

When creating custom components, add tests to the appropriate directory:

```
# For custom encoder
tests/test_encoders/test_my_custom_encoder.py

# For custom layer
tests/test_layers/test_my_custom_layer.py

# For custom node
tests/test_nodes/test_my_custom_node.py

# For experimental feature
tests/test_experimental/test_new_feature.py
```

### Basic Test Template

```python
import pytest
import torch
from difflut.nodes import MyCustomNode
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY


class TestMyCustomNode:
    """Test suite for MyCustomNode component."""
    
    @pytest.fixture
    def node_config(self):
        """Fixture providing standard node configuration."""
        return NodeConfig(input_dim=6, output_dim=1)
    
    @pytest.fixture
    def node(self, node_config):
        """Fixture providing initialized node."""
        return MyCustomNode(**node_config.to_dict())
    
    def test_initialization(self, node):
        """Test node initializes with correct shape."""
        assert node.input_dim == 6
        assert node.output_dim == 1
    
    def test_forward_pass_shape(self, node):
        """Test forward pass produces correct output shape."""
        x = torch.randn(32, 6)
        output = node(x)
        assert output.shape == (32, 1)
    
    def test_forward_pass_differentiable(self, node):
        """Test gradients flow through forward pass."""
        x = torch.randn(32, 6, requires_grad=True)
        output = node(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
```

### Testing Patterns for Different Component Types

#### Testing Nodes

```python
import pytest
import torch
from difflut.nodes import LinearLUTNode
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY


class TestCustomNode:
    """Test custom node implementation."""
    
    def test_node_registration(self):
        """Test node is properly registered."""
        assert 'my_custom_node' in REGISTRY.list_nodes()
    
    def test_node_instantiation_from_registry(self):
        """Test creating node via registry."""
        NodeClass = REGISTRY.get_node('my_custom_node')
        config = NodeConfig(input_dim=4, output_dim=1)
        node = NodeClass(**config.to_dict())
        assert node is not None
    
    def test_batch_processing(self):
        """Test processing different batch sizes."""
        config = NodeConfig(input_dim=4, output_dim=1)
        node = LinearLUTNode(**config.to_dict())
        
        for batch_size in [1, 16, 32, 128]:
            x = torch.randn(batch_size, 4)
            output = node(x)
            assert output.shape == (batch_size, 1)
    
    def test_gradient_flow(self):
        """Test gradients flow through computation."""
        config = NodeConfig(input_dim=4, output_dim=1)
        node = LinearLUTNode(**config.to_dict())
        
        x = torch.randn(32, 4, requires_grad=True)
        output = node(x)
        loss = output.mean()
        loss.backward()
        
        assert x.grad is not None
        for param in node.parameters():
            assert param.grad is not None
    
    def test_with_initializer(self):
        """Test node with custom initializer."""
        init_fn = REGISTRY.get_initializer('kaiming_normal')
        config = NodeConfig(
            input_dim=4,
            output_dim=1,
            init_fn=init_fn,
            init_kwargs={'a': 0.0, 'mode': 'fan_in'}
        )
        node = LinearLUTNode(**config.to_dict())
        assert node is not None
    
    def test_with_regularizer(self):
        """Test node with regularization."""
        reg_fn = REGISTRY.get_regularizer('l2')
        config = NodeConfig(
            input_dim=4,
            output_dim=1,
            regularizers={'l2': reg_fn}
        )
        node = LinearLUTNode(**config.to_dict())
        reg_loss = node.regularization()
        assert reg_loss >= 0
```

#### Testing Encoders

```python
import pytest
import torch
from difflut.encoder import MyCustomEncoder


class TestCustomEncoder:
    """Test custom encoder implementation."""
    
    def test_encoder_registration(self):
        """Test encoder is registered."""
        assert 'my_custom_encoder' in REGISTRY.list_encoders()
    
    def test_fit_on_data(self):
        """Test encoder fitting."""
        encoder = MyCustomEncoder(num_bits=8)
        train_data = torch.randn(1000, 100)
        encoder.fit(train_data)
        # Verify encoder stored statistics
        assert hasattr(encoder, 'min_val')
        assert hasattr(encoder, 'max_val')
    
    def test_encode_shape(self):
        """Test encoding produces correct shape."""
        encoder = MyCustomEncoder(num_bits=8, flatten=True)
        train_data = torch.randn(1000, 100)
        encoder.fit(train_data)
        
        x = torch.randn(32, 100)
        encoded = encoder(x)
        
        # With flatten=True: (batch, features * bits)
        assert encoded.shape == (32, 100 * 8)
    
    def test_encode_shape_no_flatten(self):
        """Test encoding without flattening."""
        encoder = MyCustomEncoder(num_bits=8, flatten=False)
        train_data = torch.randn(1000, 100)
        encoder.fit(train_data)
        
        x = torch.randn(32, 100)
        encoded = encoder(x)
        
        # With flatten=False: (batch, features, bits)
        assert encoded.shape == (32, 100, 8)
    
    def test_encode_output_range(self):
        """Test encoded values are in valid range."""
        encoder = MyCustomEncoder(num_bits=8)
        train_data = torch.randn(1000, 100)
        encoder.fit(train_data)
        
        x = torch.randn(32, 100)
        encoded = encoder(x)
        
        # Typically binary or probability range
        assert encoded.min() >= 0
        assert encoded.max() <= 1
```

#### Testing Layers

```python
import pytest
import torch
from difflut.layers import RandomLayer
from difflut.nodes import LinearLUTNode
from difflut.nodes.node_config import NodeConfig
from difflut.layers.layer_config import LayerConfig


class TestCustomLayer:
    """Test custom layer implementation."""
    
    def test_layer_registration(self):
        """Test layer is registered."""
        assert 'my_custom_layer' in REGISTRY.list_layers()
    
    def test_layer_instantiation(self):
        """Test creating layer."""
        node_config = NodeConfig(input_dim=4, output_dim=1)
        layer = RandomLayer(
            input_size=100,
            output_size=32,
            node_type=LinearLUTNode,
            n=4,
            node_kwargs=node_config
        )
        assert layer is not None
    
    def test_layer_forward_shape(self):
        """Test layer forward pass shape."""
        node_config = NodeConfig(input_dim=4, output_dim=1)
        layer = RandomLayer(
            input_size=100,
            output_size=32,
            node_type=LinearLUTNode,
            n=4,
            node_kwargs=node_config
        )
        
        x = torch.randn(16, 100)
        output = layer(x)
        
        # Output shape: (batch, output_size * output_dim_per_node)
        assert output.shape == (16, 32)
    
    def test_layer_with_bit_flipping(self):
        """Test layer with bit flip augmentation."""
        layer_config = LayerConfig(flip_probability=0.1)
        node_config = NodeConfig(input_dim=4, output_dim=1)
        
        layer = RandomLayer(
            input_size=100,
            output_size=32,
            node_type=LinearLUTNode,
            n=4,
            node_kwargs=node_config,
            layer_config=layer_config
        )
        
        x = torch.randn(16, 100)
        output = layer(x)
        assert output.shape == (16, 32)
    
    def test_layer_with_gradient_stabilization(self):
        """Test layer with gradient stabilization."""
        layer_config = LayerConfig(
            grad_stabilization='layerwise',
            grad_target_std=1.0
        )
        node_config = NodeConfig(input_dim=4, output_dim=1)
        
        layer = RandomLayer(
            input_size=100,
            output_size=32,
            node_type=LinearLUTNode,
            n=4,
            node_kwargs=node_config,
            layer_config=layer_config
        )
        
        x = torch.randn(16, 100, requires_grad=True)
        output = layer(x)
        loss = output.mean()
        loss.backward()
        
        assert x.grad is not None
    
    @pytest.mark.training
    def test_layer_learning(self):
        """Test layer can learn from data (training test)."""
        node_config = NodeConfig(input_dim=4, output_dim=1)
        layer = RandomLayer(
            input_size=100,
            output_size=32,
            node_type=LinearLUTNode,
            n=4,
            node_kwargs=node_config
        )
        
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        
        # Simple synthetic dataset
        x = torch.randn(100, 100)
        y = torch.randn(100, 32)
        
        initial_loss = None
        for epoch in range(10):
            optimizer.zero_grad()
            output = layer(x)
            loss = ((output - y) ** 2).mean()
            
            if epoch == 0:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        
        # Verify that loss decreased during training
        assert final_loss < initial_loss, (
            f"Layer did not learn: initial_loss={initial_loss:.4f}, "
            f"final_loss={final_loss:.4f}"
        )
```

### Experimental Features

For new features under development, use the `@pytest.mark.experimental` marker:

```python
import pytest
from difflut.experimental import NewFeature


@pytest.mark.experimental
class TestNewFeature:
    """Test suite for experimental new feature."""
    
    @pytest.mark.slow
    def test_new_feature_basic(self):
        """Test basic new feature functionality."""
        feature = NewFeature()
        result = feature.compute(torch.randn(10, 10))
        assert result is not None
    
    @pytest.mark.slow
    def test_new_feature_edge_cases(self):
        """Test edge cases in new feature."""
        feature = NewFeature()
        
        # Test with empty input
        with pytest.raises(ValueError):
            feature.compute(torch.tensor([]))
        
        # Test with single element
        result = feature.compute(torch.randn(1, 1))
        assert result.shape == (1, 1)
```

Then run experimental tests:

```bash
# Test new features
pytest tests/test_experimental/ -v -m experimental

# Run experimental tests locally during development
pytest tests/ -v -m "experimental and not slow"
```

---

## Testing Utilities

### conftest.py and Fixtures

Common fixtures are provided in `tests/conftest.py`:

```python
import pytest
import torch


@pytest.fixture
def device():
    """Fixture providing appropriate device (CPU or CUDA)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def simple_batch():
    """Fixture providing simple batch of data."""
    return torch.randn(32, 10)


@pytest.fixture
def small_model(device):
    """Fixture providing small test model."""
    # ... create small model ...
    pass
```

### Using Fixtures

```python
class TestMyComponent:
    def test_with_device(self, device):
        """Test using device fixture."""
        x = torch.randn(10, 10, device=device)
        # ... test code ...
    
    def test_with_batch(self, simple_batch):
        """Test using batch fixture."""
        assert simple_batch.shape == (32, 10)
    
    def test_with_multiple_fixtures(self, device, small_model):
        """Test using multiple fixtures."""
        model = small_model.to(device)
        output = model(torch.randn(10, 10, device=device))
```

### testing_utils.py

Helper functions for common test operations:

```python
# tests/testing_utils.py

def assert_tensor_close(actual, expected, rtol=1e-5, atol=1e-8):
    """Assert two tensors are close."""
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol)


def create_test_batch(batch_size, features, device='cpu'):
    """Create test batch."""
    return torch.randn(batch_size, features, device=device)


def check_gradients_flow(model, input_tensor):
    """Verify gradients flow through model."""
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()
    
    for param in model.parameters():
        assert param.grad is not None
```

## Resources

- **Pytest Documentation**: https://docs.pytest.org/
- **Test Configuration**: `pyproject.toml` (pytest section)
- **Test Utilities**: `tests/testing_utils.py`
- **Fixtures**: `tests/conftest.py`
- **Example Tests**: `tests/test_nodes/`, `tests/test_layers/`, `tests/test_encoders/`
- **CI Workflow**: `.github/workflows/tests.yml`

