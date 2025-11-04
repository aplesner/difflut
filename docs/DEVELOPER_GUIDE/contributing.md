# Contributing to DiffLUT

Thank you for your interest in contributing to DiffLUT! This guide explains the development process, testing requirements, and how to submit pull requests.

## Getting Started

### 1. Fork and Clone

```bash
# Fork repository on GitLab
git clone https://gitlab.ethz.ch/your-username/difflut.git
cd difflut/difflut

# Add upstream remote
git remote add upstream https://gitlab.ethz.ch/disco-students/hs25/difflut.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install additional dev tools
pip install pytest pytest-cov black flake8 mypy
```

### 3. Create Feature Branch

```bash
# Update local repository
git fetch upstream
git checkout upstream/main

# Create feature branch
git checkout -b feature/my-new-feature
```

## Development Workflow

### Type Hints (PEP 484–compatible)

All code uses PEP 484–compatible type annotations for full static type checker support (mypy) and Python < 3.10 compatibility. Use `Optional[T]` instead of `T | None`:

```python
from typing import Optional, Tuple, Dict, Any, Callable
import torch

def process_input(
    x: torch.Tensor,
    num_bits: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process input with PEP 484-compatible type hints.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    num_bits : Optional[int]
        Number of bits (optional)
    config : Optional[Dict[str, Any]]
        Configuration dictionary
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Processed output
    """
    return x, x
```

✓ Use `Optional[T]` for optional parameters  
✓ Use `Dict[K, V]` for dictionaries  
✓ Use `Callable[[ArgTypes], ReturnType]` for functions  
✗ Avoid `T | None` syntax (Python 3.10+ only)

### Default Constants & Warnings

All modules define default configuration values as module-level constants and use a unified warning system when defaults are applied.

**Pattern for default constants** at top of each module:

```python
# Node-specific defaults
DEFAULT_NODE_INPUT_DIM: int = 6
DEFAULT_NODE_OUTPUT_DIM: int = 1
DEFAULT_NODE_LAYER_SIZE: int = 1
DEFAULT_NODE_USE_CUDA: bool = True

# Warning thresholds
NODE_INPUT_DIM_WARNING_THRESHOLD: int = 10
```

**Using defaults in constructors** with explicit warnings:

```python
from difflut.utils.warnings import warn_default_value

class MyNode(BaseNode):
    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        use_cuda: bool = DEFAULT_NODE_USE_CUDA,
        ...
    ) -> None:
        # Apply defaults with warnings for None parameters
        if input_dim is None:
            self.input_dim = DEFAULT_NODE_INPUT_DIM
            warn_default_value("input_dim", self.input_dim, stacklevel=2)
        else:
            self.input_dim = input_dim
        
        # Warn if value exceeds threshold
        if self.input_dim > NODE_INPUT_DIM_WARNING_THRESHOLD:
            warn_default_value(
                f"input_dim ({self.input_dim}) exceeds recommended threshold",
                NODE_INPUT_DIM_WARNING_THRESHOLD,
                stacklevel=2
            )
```

**Benefits:**
- Centralized default management
- Traceable parameter application
- Consistent warning behavior across modules
- Easy to adjust defaults in one place

### NodeConfig for Type-Safe Configuration

Instead of passing raw `node_kwargs: dict`, use the typed `NodeConfig` class for clean parameter passing:

```python
from difflut.nodes.node_config import NodeConfig
from difflut.nodes import LinearLUTNode
from difflut.layers import RandomLayer

# Old way (discouraged)
node_kwargs = {'input_dim': 6, 'output_dim': 1}

# New way (recommended)
config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=my_init_fn,
    init_kwargs={'scale': 0.1},
    extra_params={'use_cuda': True}
)

# Use in layer
layer = RandomLayer(
    input_size=100,
    output_size=50,
    node_type=LinearLUTNode,
    node_kwargs=config
)
```

**NodeConfig advantages:**
- Type-safe: IDE autocompletion and mypy checking
- Self-documenting: All parameters visible and typed
- Extensible: `extra_params` dict for node-specific options
- Maintainable: Single source of truth for parameter docs

See `difflut/nodes/node_config.py` for full documentation and examples.

### Architecture: 2D Tensors with nn.ModuleList

**Current Architecture** (as of November 2025):
- **Nodes process 2D tensors**: `(batch_size, input_dim)` → `(batch_size, output_dim)`
- **Layers use nn.ModuleList**: Each layer creates `output_size` independent node instances
- **No layer_size dimension**: Removed for simplicity and PyTorch best practices
- **Independent nodes**: Each node in ModuleList processes batches independently

```python
class MyLayer(BaseLUTLayer):
    def __init__(self, input_size, output_size, node_type, n, node_kwargs):
        super().__init__(...)
        
        # Create output_size independent nodes
        self.nodes = nn.ModuleList([
            node_type(**node_kwargs) for _ in range(output_size)
        ])
    
    def forward(self, x):
        # x: (batch, input_size)
        outputs = []
        for i, node in enumerate(self.nodes):
            # Map inputs for this node: (batch, input_size) → (batch, n)
            node_inputs = self.mapping(x, i)  # (batch, n)
            # Process through node: (batch, n) → (batch, 1)
            node_output = node(node_inputs)
            outputs.append(node_output)
        # Concatenate: list of (batch, 1) → (batch, output_size)
        return torch.cat(outputs, dim=1)
```

**Key points for contributors:**
- All nodes must handle 2D tensors only
- Parameters like `layer_size` are kept for backward compatibility but unused
- Layers iterate through `self.nodes` ModuleList
- Each node processes independently - no shared state across nodes in a layer

### Code Style

We follow PEP 8 with a few conventions:

**Format code with Black**:

```bash
black difflut/ tests/ examples/
```

**Lint with Flake8**:

```bash
flake8 difflut/ tests/ --max-line-length=100
```

**Type hints** (optional but encouraged):

```python
from typing import Tuple
import torch

def process_input(x: torch.Tensor, num_bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process input with type hints."""
    return x, x
```

### Docstring Format

Use NumPy-style docstrings:

```python
def my_function(x, param=None):
    """
    Brief description of function.
    
    Longer description if needed. Explain the algorithm,
    any important details, and usage patterns.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch, features)
    param : int, optional
        Parameter description. Default is None.
    
    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch, outputs)
    
    Raises
    ------
    ValueError
        If input dimensions don't match expected
    
    Examples
    --------
    >>> x = torch.randn(32, 10)
    >>> output = my_function(x, param=5)
    >>> print(output.shape)
    torch.Size([32, 1])
    """
    pass
```

### Commits

Write clear commit messages:

```bash
# Bad
git commit -m "fix stuff"

# Good
git commit -m "Fix gradient computation in CustomNode

- Correct tensor indexing in backward pass
- Add validation for input shapes
- Includes test coverage"
```

## Testing

### Writing Tests

Place tests in `tests/` directory:

```python
# tests/test_my_component.py
import pytest
import torch
from difflut.nodes import MyCustomNode

class TestMyCustomNode:
    """Test suite for MyCustomNode."""
    
    def test_initialization(self):
        """Test node initialization."""
        node = MyCustomNode(input_dim=[4], output_dim=[1])
        assert node.num_inputs == 4
        assert node.num_outputs == 1
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        node = MyCustomNode(input_dim=[4], output_dim=[2])
        x = torch.randint(0, 2, (32, 4))
        output = node(x)
        assert output.shape == (32, 2)
    
    def test_forward_backward(self):
        """Test gradient computation."""
        node = MyCustomNode(input_dim=[4], output_dim=[1])
        x = torch.randint(0, 2, (16, 4), dtype=torch.float32)
        
        output = node(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        assert node.weights.grad is not None
        assert torch.all(torch.isfinite(node.weights.grad))
    
    def test_training_vs_eval(self):
        """Test difference between training and eval modes."""
        node = MyCustomNode(input_dim=[4], output_dim=[1])
        x = torch.randint(0, 2, (32, 4))
        
        node.train()
        train_output = node(x)
        
        node.eval()
        eval_output = node(x)
        
        # Should be same for this node (if not using dropout, etc)
        assert torch.allclose(train_output, eval_output)
    
    def test_regularization(self):
        """Test regularization computation."""
        node = MyCustomNode(input_dim=[4], output_dim=[1])
        reg = node.regularization()
        
        assert isinstance(reg, torch.Tensor)
        assert reg.ndim == 0  # Scalar
        assert reg.item() >= 0
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_my_component.py

# Run specific test class
pytest tests/test_my_component.py::TestMyCustomNode

# Run with verbose output
pytest -v tests/

# Run with coverage report
pytest --cov=difflut tests/
```

### Test Coverage Requirements

- Aim for >80% code coverage
- All public methods should have tests
- Test both success and error cases
- Test edge cases (empty inputs, single sample, etc.)

```bash
# Generate coverage report
pytest --cov=difflut --cov-report=html tests/
# Open htmlcov/index.html in browser
```

## Documentation

### Docstrings

All public functions/classes need docstrings:

```python
from difflut.nodes import BaseNode

class MyNode(BaseNode):
    """
    My custom LUT node implementation.
    
    This node implements a novel computation strategy that combines
    table lookup with polynomial approximation for improved accuracy.
    
    Parameters
    ----------
    input_dim : list
        [n] - Number of binary inputs
    output_dim : list
        [m] - Number of outputs
    degree : int, optional
        Polynomial degree. Default is 3.
    
    Attributes
    ----------
    weights : torch.nn.Parameter
        LUT weights of shape (2^n, m)
    
    Examples
    --------
    >>> node = MyNode(input_dim=[4], output_dim=[1], degree=3)
    >>> x = torch.randint(0, 2, (32, 4))
    >>> output = node(x)
    >>> print(output.shape)
    torch.Size([32, 1])
    """
    pass
```

### README Updates

If adding major features, update relevant documentation:

- `README.md` - Project overview (keep short)
- `docs/USER_GUIDE.md` - User-facing features
- `docs/DEVELOPER_GUIDE.md` - Developer features

### Code Comments

Use comments for complex logic:

```python
# Convert binary input to table indices
# Example: [0, 1, 1, 0] -> 0*1 + 1*2 + 1*4 + 0*8 = 6
indices = self._binary_to_index(x)
```

## Submitting Pull Requests

### Before Submitting

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests locally**:
   ```bash
   pytest tests/ -v
   ```

3. **Check code style**:
   ```bash
   black difflut/ tests/
   flake8 difflut/ tests/
   ```

4. **Update documentation** if needed

### Creating Pull Request

1. **Push to fork**:
   ```bash
   git push origin feature/my-new-feature
   ```

2. **Create PR on GitLab**:
   - Go to https://gitlab.ethz.ch/disco-students/hs25/difflut/-/merge_requests
   - Click "New merge request"
   - Select your branch

3. **Fill PR template**:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Motivation
   Why is this change needed?
   
   ## Changes
   - Change 1
   - Change 2
   
   ## Testing
   How to test the changes?
   
   ## Checklist
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] Code follows style guide
   - [ ] No breaking changes
   ```

### PR Guidelines

- **One feature per PR** - Keep PRs focused
- **Clear description** - Explain what and why
- **Test coverage** - Include tests for new code
- **Updated docs** - Update relevant documentation
- **Clean history** - Squash if needed

### Code Review

- Respond to reviewer comments
- Push additional commits if changes needed
- Don't force-push during review

## Types of Contributions

### Bug Fixes

```bash
git checkout -b bugfix/issue-description

# Fix the bug
# Add test demonstrating fix
pytest tests/
git push origin bugfix/issue-description
```

### New Features

```bash
git checkout -b feature/feature-name

# Implement feature
# Add comprehensive tests
# Update documentation
pytest tests/
git push origin feature/feature-name
```

### New Nodes/Encoders/Layers

See [Creating Components](creating_components.md) for detailed guidelines.

### Documentation

```bash
git checkout -b docs/improve-guide

# Update documentation files
# Test links and formatting
git push origin docs/improve-guide
```

## Common Issues

### Tests Fail Locally

```bash
# Clean environment
pip uninstall difflut
pip install -e ".[dev]"

# Rerun tests
pytest tests/ -v
```

### CUDA Tests Fail

Some tests may require CUDA:

```bash
# Run CPU-only tests
pytest tests/ -k "not cuda"

# Or skip GPU tests entirely
pytest tests/ -m "not gpu"
```

### Style Check Fails

```bash
# Auto-format with black
black difflut/ tests/

# Fix flake8 issues
flake8 difflut/ --show-source --statistics
```

## Development Tips

### Quick Test Loop

```bash
# Watch for changes and run tests
pytest-watch tests/

# Or manually in loop
while true; do pytest tests/my_test.py -v && sleep 2; done
```

### Debug Mode

```python
# Use pdb for debugging
import pdb

def my_function(x):
    pdb.set_trace()  # Debugger will stop here
    return x
```

Run with:
```bash
pytest tests/test_file.py --pdb
```

### Memory Profiling

```bash
pip install memory_profiler

@profile
def my_function():
    pass

python -m memory_profiler my_script.py
```

### Performance Profiling

```python
import cProfile
import pstats

cProfile.run('my_function()', 'output.prof')
p = pstats.Stats('output.prof')
p.sort_stats('cumulative').print_stats(10)
```

## Getting Help

- **Issues**: https://gitlab.ethz.ch/disco-students/hs25/difflut/-/issues
- **Email**: sbuehrer@ethz.ch
- **Discussions**: Review existing PRs for patterns

## Community Guidelines

- Be respectful and constructive
- Assume good intent
- Help others learn
- Share knowledge
- Give credit for ideas

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Treat all contributors with respect.

## License

By contributing to DiffLUT, you agree that your contributions will be licensed under the MIT License.

## Next Steps

- Read [Creating Components](creating_components.md) to implement new features
- Check [Packaging Guide](packaging.md) for distribution info
- Review existing code in `difflut/` directory
- Start with a small PR to get familiar with process

Thank you for contributing!
