# Creating Custom Nodes

Learn how to implement and register custom DiffLUT nodes with initializers and regularizers.

---

## Overview

Nodes define computation at individual LUT units. Each node:

1. **Extends** `BaseNode`
2. **Registers** using `@register_node` decorator
3. **Implements** `forward_train()` and `forward_eval()` methods
4. **Optionally adds** CUDA support for performance
5. **Supports** custom initializers and regularizers via registry

### Key Concepts

- **LUT Nodes**: Lookup table-based computation
- **Binary Indexing**: Convert continuous inputs to discrete LUT indices
- **Initializers**: Functions that initialize LUT weights
- **Regularizers**: Functions that add regularization terms
- **CUDA Kernels**: Optional GPU acceleration
- **Training vs. Eval**: Different forward passes for training and evaluation

---

## Type-Safe Configuration

DiffLUT uses `NodeConfig` for type-safe node parameters:

```python
from difflut.nodes.node_config import NodeConfig
from difflut import register_node, REGISTRY

# Module defaults (at top level)
DEFAULT_NODE_INPUT_DIM: int = 6
DEFAULT_NODE_OUTPUT_DIM: int = 1

@register_node('my_custom_node')
class MyCustomNode(BaseNode):
    """Custom node with clear documentation."""
    
    def __init__(
        self,
        input_dim: int = DEFAULT_NODE_INPUT_DIM,
        output_dim: int = DEFAULT_NODE_OUTPUT_DIM,
        init_fn=None,
        init_kwargs=None,
        regularizers=None,
        my_custom_param=None,
        **kwargs
    ) -> None:
        """
        Initialize custom node.
        
        Args:
            input_dim: Input LUT dimension (e.g., 6 for 2^6=64 table entries)
            output_dim: Output dimension
            init_fn: Initializer function (retrieved from REGISTRY)
            init_kwargs: Initializer keyword arguments
            regularizers: Dict of regularizer name -> function pairs
            my_custom_param: Custom parameter for this node type
            **kwargs: Additional parameters
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            init_fn=init_fn,
            init_kwargs=init_kwargs,
            regularizers=regularizers,
            **kwargs
        )
        self.my_custom_param = my_custom_param
```

**Why this pattern?**
- Type safety with full type hints
- Initializers/regularizers retrieved from REGISTRY (not strings)
- Clear separation of structural vs. runtime parameters
- Easy configuration via NodeConfig

---

## Initializers and Regularizers

### Using Initializers

Initializers are functions that initialize node weights:

```python
from difflut.registry import REGISTRY
from difflut.nodes.node_config import NodeConfig

# Get initializer from registry (not a string!)
init_fn = REGISTRY.get_initializer('kaiming_normal')

# Create NodeConfig with initializer function
node_config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=init_fn,  # Actual function, not string
    init_kwargs={
        'a': 0.0,
        'mode': 'fan_in',
        'nonlinearity': 'relu'
    }
)

# Use in node instantiation
node = MyCustomNode(**node_config.to_dict())
```

### Available Initializers

Common initializers in the registry:

```python
from difflut.registry import REGISTRY

print("Available initializers:", REGISTRY.list_initializers())
# Output: ['kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform', 'normal', 'uniform']

# Get any initializer
init_normal = REGISTRY.get_initializer('normal')
init_kaiming = REGISTRY.get_initializer('kaiming_normal')
```

### Using Regularizers

Regularizers are functions that add regularization terms:

```python
from difflut.registry import REGISTRY
from difflut.nodes.node_config import NodeConfig

# Get regularizers from registry (not strings!)
l2_reg = REGISTRY.get_regularizer('l2')
l1_reg = REGISTRY.get_regularizer('l1')

# Create NodeConfig with regularizers
node_config = NodeConfig(
    input_dim=6,
    output_dim=1,
    regularizers={
        'l2': l2_reg,    # Actual function, not string
        'l1': l1_reg,    # Actual function, not string
    }
)

# Use in node
node = MyCustomNode(**node_config.to_dict())

# Regularization term computed during forward pass
reg_loss = node.regularization()
```

### Available Regularizers

Common regularizers in the registry:

```python
from difflut.registry import REGISTRY

print("Available regularizers:", REGISTRY.list_regularizers())
# Output: ['l1', 'l2', 'l1_l2', 'entropy']

# Get any regularizer
reg_l2 = REGISTRY.get_regularizer('l2')
reg_l1 = REGISTRY.get_regularizer('l1')
```

### Implementing Custom Initializers

```python
from difflut import register_initializer

@register_initializer('my_init')
def my_custom_initializer(tensor, gain=1.0, scale=0.1, **kwargs):
    """Custom initialization function."""
    with torch.no_grad():
        # Custom initialization logic
        tensor.normal_(0, scale * gain)
    return tensor

# Use in nodes
init_fn = REGISTRY.get_initializer('my_init')
node_config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=init_fn,
    init_kwargs={'gain': 1.0, 'scale': 0.05}
)
```

### Implementing Custom Regularizers

```python
from difflut import register_regularizer

@register_regularizer('my_reg')
def my_custom_regularizer(node, **kwargs):
    """Custom regularization function."""
    # node is the node instance
    weight = node.get_weights()
    
    # Example: entropy regularization
    p = torch.nn.functional.softmax(weight.flatten(), dim=0)
    entropy = -torch.sum(p * torch.log(p + 1e-8))
    
    return entropy * kwargs.get('weight', 0.01)

# Use in nodes
reg_fn = REGISTRY.get_regularizer('my_reg')
node_config = NodeConfig(
    input_dim=6,
    output_dim=1,
    regularizers={'entropy': reg_fn}
)
```

---

## Complete Node Implementation

### Step 1: Define Module Defaults

```python
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
from difflut.nodes import BaseNode
from difflut.nodes.node_config import NodeConfig
from difflut import register_node

# Module-level defaults (CAPITALS convention)
DEFAULT_NODE_INPUT_DIM: int = 6
DEFAULT_NODE_OUTPUT_DIM: int = 1
DEFAULT_TEMPERATURE: float = 1.0
```

### Step 2: Implement BaseNode

```python
@register_node('probabilistic_example')
class ProbabilisticExampleNode(BaseNode):
    """
    Probabilistic LUT node with temperature scaling.
    
    Processes 2D tensors: (batch_size, input_dim) → (batch_size, output_dim)
    
    During training: Samples from learned LUT based on soft indices
    During eval: Returns deterministic expected values
    """
    
    def __init__(
        self,
        input_dim: int = DEFAULT_NODE_INPUT_DIM,
        output_dim: int = DEFAULT_NODE_OUTPUT_DIM,
        init_fn: Optional[Callable] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        regularizers: Optional[Dict[str, Callable]] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs
    ) -> None:
        """
        Initialize probabilistic node.
        
        Args:
            input_dim: Input LUT dimension (e.g., 6 for 2^6=64 entries)
            output_dim: Output dimension
            init_fn: Initializer function (from REGISTRY)
            init_kwargs: Arguments for initializer
            regularizers: Dict of name -> regularizer function (from REGISTRY)
            temperature: Temperature for softmax (higher = softer)
            **kwargs: Additional parameters
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            init_fn=init_fn,
            init_kwargs=init_kwargs,
            regularizers=regularizers,
            **kwargs
        )
        
        self.temperature = temperature
        
        # Initialize LUT weights
        lut_size = 2 ** input_dim
        self.weights = nn.Parameter(
            torch.randn(lut_size, output_dim)
        )
        
        # Apply initializer if provided
        if init_fn is not None:
            init_kwargs = init_kwargs or {}
            init_fn(self.weights, **init_kwargs)
    
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass with probabilistic sampling.
        
        Args:
            x: Binary input of shape (batch_size, input_dim)
        
        Returns:
            Output of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        
        # Convert binary input to float indices
        # x is binary [0,1], compute index = sum(x_i * 2^i)
        powers = 2 ** torch.arange(self.input_dim, device=x.device)
        indices_float = torch.sum(x * powers, dim=1)  # (batch,)
        
        # Create soft indices with temperature
        # Add small noise for gradient flow during training
        noise = torch.randn_like(indices_float) * 0.1
        soft_indices = indices_float + noise
        soft_indices = torch.clamp(soft_indices, 0, 2**self.input_dim - 1)
        
        # Use soft indices for soft indexing into LUT
        normalized_indices = soft_indices / (2**self.input_dim - 1)
        
        # Simple nearest-neighbor lookup (can use soft attention instead)
        hard_indices = indices_float.long()
        output = self.weights[hard_indices]  # (batch, output_dim)
        
        return output
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Eval forward pass - deterministic.
        
        Args:
            x: Binary input of shape (batch_size, input_dim)
        
        Returns:
            Output of shape (batch_size, output_dim)
        """
        # Convert binary input to indices
        powers = 2 ** torch.arange(self.input_dim, device=x.device)
        indices = torch.sum(x * powers, dim=1).long()  # (batch,)
        
        # Deterministic lookup
        output = self.weights[indices]
        
        return output
    
    def regularization(self) -> torch.Tensor:
        """
        Compute regularization term.
        
        Returns:
            Scalar regularization loss
        """
        total_reg = torch.tensor(0.0, device=self.weights.device)
        
        # Apply each registered regularizer
        if self.regularizers is not None:
            for reg_name, reg_fn in self.regularizers.items():
                reg_loss = reg_fn(self, weight=0.01)
                total_reg = total_reg + reg_loss
        
        return total_reg
    
    def export_bitstream(self) -> Any:
        """
        Export node as bitstream (for FPGA deployment).
        
        Returns:
            Exported representation (usually numpy array)
        """
        return self.weights.detach().cpu().numpy()
```

### Step 3: Use the Node

```python
from difflut.registry import REGISTRY

# Get initializer and regularizer from registry
init_fn = REGISTRY.get_initializer('kaiming_normal')
l2_reg = REGISTRY.get_regularizer('l2')

# Create NodeConfig
node_config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=init_fn,
    init_kwargs={'a': 0.0, 'mode': 'fan_in'},
    regularizers={'l2': l2_reg}
)

# Create node
node = ProbabilisticExampleNode(**node_config.to_dict(), temperature=1.0)

# Use in forward pass
x = torch.randint(0, 2, (32, 6)).float()  # Binary input (batch=32, dim=6)
y_train = node.forward_train(x)
y_eval = node.forward_eval(x)
reg_loss = node.regularization()
```

---

## Advanced Examples

### Example 1: Learnable LUT Node with Softmax

```python
@register_node('learnable_lut')
class LearnableLUTNode(BaseNode):
    """LUT with learnable connection weights between input and LUT entry."""
    
    def __init__(
        self,
        input_dim: int = DEFAULT_NODE_INPUT_DIM,
        output_dim: int = DEFAULT_NODE_OUTPUT_DIM,
        init_fn: Optional[Callable] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        regularizers: Optional[Dict[str, Callable]] = None,
        **kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            init_fn=init_fn,
            init_kwargs=init_kwargs,
            regularizers=regularizers,
            **kwargs
        )
        
        # Learnable input-to-LUT connection matrix
        self.mapping = nn.Parameter(
            torch.randn(input_dim, 2**input_dim)
        )
        
        # LUT values
        lut_size = 2 ** input_dim
        self.weights = nn.Parameter(torch.randn(lut_size, output_dim))
        
        if init_fn:
            init_kwargs = init_kwargs or {}
            init_fn(self.weights, **init_kwargs)
    
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """Soft indexing via learnable mapping."""
        # x: (batch, input_dim)
        
        # Compute soft indices using mapping matrix
        logits = torch.matmul(x, self.mapping)  # (batch, 2^input_dim)
        soft_indices = torch.softmax(logits, dim=1)  # (batch, 2^input_dim)
        
        # Weighted sum over LUT entries
        output = torch.matmul(soft_indices, self.weights)  # (batch, output_dim)
        
        return output
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Hard indexing - argmax during evaluation."""
        # x: (batch, input_dim) - assumed binary
        
        powers = 2 ** torch.arange(self.input_dim, device=x.device)
        indices = torch.sum(x * powers, dim=1).long()
        
        return self.weights[indices]
    
    def regularization(self) -> torch.Tensor:
        """Regularize mapping matrix."""
        total_reg = torch.tensor(0.0, device=self.weights.device)
        
        # L2 on weights
        total_reg += 0.001 * torch.sum(self.weights ** 2)
        
        # L2 on mapping
        total_reg += 0.0001 * torch.sum(self.mapping ** 2)
        
        return total_reg
    
    def export_bitstream(self) -> Any:
        """Export as tuple (mapping, weights)."""
        return (
            self.mapping.detach().cpu().numpy(),
            self.weights.detach().cpu().numpy()
        )
```

### Example 2: Polynomial Approximation Node

```python
@register_node('polynomial')
class PolynomialNode(BaseNode):
    """Node using polynomial basis functions instead of direct LUT."""
    
    def __init__(
        self,
        input_dim: int = DEFAULT_NODE_INPUT_DIM,
        output_dim: int = DEFAULT_NODE_OUTPUT_DIM,
        degree: int = 3,
        init_fn: Optional[Callable] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        regularizers: Optional[Dict[str, Callable]] = None,
        **kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            init_fn=init_fn,
            init_kwargs=init_kwargs,
            regularizers=regularizers,
            **kwargs
        )
        
        self.degree = degree
        
        # Polynomial coefficients: (degree+1, input_dim, output_dim)
        self.coeffs = nn.Parameter(
            torch.randn(degree + 1, input_dim, output_dim)
        )
        
        if init_fn:
            init_kwargs = init_kwargs or {}
            for i in range(degree + 1):
                init_fn(self.coeffs[i], **init_kwargs)
    
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """Polynomial approximation."""
        # x: (batch, input_dim)
        output = torch.zeros(x.shape[0], self.output_dim, device=x.device)
        
        for d in range(self.degree + 1):
            # Compute x^d
            x_power = x ** d  # (batch, input_dim)
            
            # Apply coefficients
            term = torch.sum(
                x_power.unsqueeze(2) * self.coeffs[d].unsqueeze(0),
                dim=1
            )  # (batch, output_dim)
            
            output = output + term
        
        return output
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_train(x)
    
    def regularization(self) -> torch.Tensor:
        return 0.001 * torch.sum(self.coeffs ** 2)
    
    def export_bitstream(self) -> Any:
        return self.coeffs.detach().cpu().numpy()
```

---

## Testing Your Node

Create `tests/test_nodes/test_my_node.py`:

```python
import torch
import pytest
from difflut.nodes import MyCustomNode
from difflut.nodes.node_config import NodeConfig
from difflut.registry import REGISTRY

class TestMyCustomNode:
    
    def test_registration(self):
        """Verify node is registered."""
        node_class = REGISTRY.get_node('my_custom_node')
        assert node_class is not None
    
    def test_initialization(self):
        """Test node initialization."""
        node = MyCustomNode(input_dim=6, output_dim=1)
        assert node.input_dim == 6
        assert node.output_dim == 1
    
    def test_forward_train_shape(self):
        """Test forward_train output shape."""
        node = MyCustomNode(input_dim=6, output_dim=2)
        x = torch.randint(0, 2, (32, 6)).float()  # Binary input
        
        y = node.forward_train(x)
        assert y.shape == (32, 2)
    
    def test_forward_eval_shape(self):
        """Test forward_eval output shape."""
        node = MyCustomNode(input_dim=6, output_dim=2)
        x = torch.randint(0, 2, (32, 6)).float()
        
        y = node.forward_eval(x)
        assert y.shape == (32, 2)
    
    def test_with_initializer(self):
        """Test node with custom initializer."""
        init_fn = REGISTRY.get_initializer('kaiming_normal')
        
        node = MyCustomNode(
            input_dim=6,
            output_dim=1,
            init_fn=init_fn,
            init_kwargs={'a': 0.0, 'mode': 'fan_in'}
        )
        
        x = torch.randint(0, 2, (32, 6)).float()
        y = node.forward_train(x)
        assert y.shape == (32, 1)
    
    def test_with_regularizer(self):
        """Test node with regularizer."""
        l2_reg = REGISTRY.get_regularizer('l2')
        
        node = MyCustomNode(
            input_dim=6,
            output_dim=1,
            regularizers={'l2': l2_reg}
        )
        
        reg_loss = node.regularization()
        assert reg_loss.item() >= 0
    
    def test_deterministic_indices(self):
        """Test that eval uses deterministic indices."""
        node = MyCustomNode(input_dim=4, output_dim=1)
        
        # Create specific binary input
        x = torch.tensor([[1, 0, 1, 0]]).float()  # index = 0*1 + 1*2 + 0*4 + 1*8 = 10
        
        y1 = node.forward_eval(x)
        y2 = node.forward_eval(x)
        
        # Should be identical
        assert torch.allclose(y1, y2)
    
    def test_device_transfer(self, device):
        """Test node works after device transfer."""
        node = MyCustomNode(input_dim=6, output_dim=1).to(device)
        x = torch.randint(0, 2, (32, 6), device=device).float()
        
        y = node.forward_train(x)
        assert y.device == device
```

---

## Key Patterns

1. **Module-Level Defaults**: Use CAPITALS for module constants
2. **Type Hints**: Full PEP 484 type hints for IDE support
3. **Registry Retrieval**: Always get initializers/regularizers from REGISTRY
4. **Binary Input**: Assume input `x` is binary [0,1]
5. **Two Forward Passes**: Separate `forward_train()` and `forward_eval()`
6. **Regularization**: Implement `regularization()` for loss term
7. **Bitstream Export**: `export_bitstream()` for deployment
8. **Docstrings**: NumPy format with all parameters

---

## Next Steps

1. **Review existing nodes** in `difflut/nodes/` for patterns
2. **Implement forward_train()** for training
3. **Implement forward_eval()** for deterministic evaluation
4. **Add initializer support** via NodeConfig
5. **Add regularizer support** via NodeConfig
6. **Add comprehensive tests** following test patterns
7. **Consider CUDA** for performance-critical nodes
8. **Integrate with pipelines** via REGISTRY

---

## Resources

- **Base Class**: `difflut/nodes/base_node.py`
- **Config Class**: `difflut/nodes/node_config.py`
- **Initializers**: `difflut/nodes/utils/initializers.py`
- **Regularizers**: `difflut/nodes/utils/regularizers.py`
- **Examples**: `difflut/nodes/probabilistic_node.py`, `difflut/nodes/linear_lut_node.py`
- **Tests**: `tests/test_nodes/`
- **User Guide**: [Nodes Guide](../../USER_GUIDE/components/nodes.md)
- **Registry**: [Registry & Pipeline Guide](../../USER_GUIDE/registry_pipeline.md)
