# Creating Custom Layers

Learn how to implement and register custom DiffLUT layers with gradient stabilization and advanced training techniques.

---

## Overview

Layers define connectivity patterns between encoded inputs and LUT nodes. All custom layers:

1. **Extend** `BaseLUTLayer`
2. **Register** using `@register_layer` decorator
3. **Implement** `forward()` method
4. **Use** `LayerConfig` for training parameters
5. **Support** gradient stabilization and bit flipping

### Key Concepts

- **Connectivity**: How inputs route to nodes (random, learnable, feature-wise, etc.)
- **Training Augmentation**: Bit flipping, gradient stabilization applied during training
- **LayerConfig**: Type-safe training parameters (not structural)
- **Node Composition**: Layers manage many nodes with consistent configuration
- **Gradient Stabilization**: Normalize gradients for stable training

---

## Type-Safe Configuration

DiffLUT uses `LayerConfig` for type-safe layer training parameters:

```python
from difflut.layers.layer_config import LayerConfig
from difflut.layers import BaseLUTLayer
from difflut import register_layer

# Module defaults (at top level)
DEFAULT_FLIP_PROBABILITY: float = 0.0
DEFAULT_GRAD_STABILIZATION: str = 'none'
DEFAULT_GRAD_TARGET_STD: float = 1.0

@register_layer('my_custom_layer')
class MyCustomLayer(BaseLUTLayer):
    """Custom layer with training augmentation."""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        node_type: str,
        n: int,
        node_kwargs=None,
        layer_config: LayerConfig = None,
        **kwargs
    ):
        """
        Initialize custom layer.
        
        Args:
            input_size: Input dimension (after encoding)
            output_size: Output dimension
            node_type: Node type from registry
            n: Input dimension for each node (node fan-in)
            node_kwargs: Parameters for node creation
            layer_config: LayerConfig with training augmentation settings
            **kwargs: Additional parameters
        """
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            node_type=node_type,
            n=n,
            node_kwargs=node_kwargs,
            layer_config=layer_config,
            **kwargs
        )
```

**Why this pattern?**
- LayerConfig handles training parameters (flip_probability, grad_stabilization)
- Structural parameters (input_size, output_size, node_type) stay in __init__
- Clear separation of architecture from training behavior
- Layer training configs don't require model retraining

---

## LayerConfig Parameters

`LayerConfig` provides training augmentation parameters:

```python
from difflut.layers.layer_config import LayerConfig

# Create training config
layer_config = LayerConfig(
    flip_probability=0.05,           # Flip 5% of bits during training
    grad_stabilization='layerwise',  # Normalize gradients
    grad_target_std=1.0,             # Target gradient std
    grad_subtract_mean=False,        # Don't subtract mean
    grad_epsilon=1e-8,               # Numerical stability
)

# Use in layer
layer = MyCustomLayer(
    input_size=512,
    output_size=256,
    node_type='probabilistic',
    n=6,
    layer_config=layer_config
)
```

### Gradient Stabilization Options

- **'none'**: No gradient stabilization
- **'layerwise'**: Normalize across entire layer (all nodes)
- **'nodewise'**: Normalize per-node separately
- **'batch'**: Normalize per-batch element

---

## Complete Layer Implementation

### Step 1: Define Module Defaults

```python
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from difflut.layers import BaseLUTLayer
from difflut.layers.layer_config import LayerConfig
from difflut.registry import REGISTRY
from difflut import register_layer

# Module-level defaults (CAPITALS convention)
DEFAULT_INPUT_SIZE: int = 512
DEFAULT_OUTPUT_SIZE: int = 256
DEFAULT_NODE_TYPE: str = 'probabilistic'
DEFAULT_INPUT_DIM: int = 6  # n - LUT input dimension

DEFAULT_FLIP_PROBABILITY: float = 0.0
DEFAULT_GRAD_STABILIZATION: str = 'none'
DEFAULT_GRAD_TARGET_STD: float = 1.0
DEFAULT_GRAD_EPSILON: float = 1e-8
```

### Step 2: Implement BaseLUTLayer

```python
@register_layer('random_routing')
class RandomRoutingLayer(BaseLUTLayer):
    """
    Layer with random fixed connectivity between inputs and nodes.
    
    Each node receives a random subset of input bits.
    Connectivity is fixed during training (not learned).
    """
    
    def __init__(
        self,
        input_size: int = DEFAULT_INPUT_SIZE,
        output_size: int = DEFAULT_OUTPUT_SIZE,
        node_type: str = DEFAULT_NODE_TYPE,
        n: int = DEFAULT_INPUT_DIM,
        node_kwargs: Optional[Dict[str, Any]] = None,
        layer_config: Optional[LayerConfig] = None,
        seed: int = 42,
        **kwargs
    ) -> None:
        """
        Initialize random routing layer.
        
        Args:
            input_size: Input dimension (e.g., 512 bits)
            output_size: Number of nodes (output dimension)
            node_type: Node type from registry (e.g., 'probabilistic')
            n: Input dimension per node (e.g., 6 bits per node)
            node_kwargs: Parameters passed to node creation
            layer_config: LayerConfig with training augmentation
            seed: Random seed for reproducible connectivity
            **kwargs: Additional parameters
        """
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            node_type=node_type,
            n=n,
            node_kwargs=node_kwargs,
            layer_config=layer_config,
            **kwargs
        )
        
        # Create random connectivity routing
        self.seed = seed
        torch.manual_seed(seed)
        
        # For each output node, select n random input indices
        self.register_buffer(
            'routing',
            torch.stack([
                torch.randperm(input_size)[:n]
                for _ in range(output_size)
            ])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with random routing.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size = x.shape[0]
        
        # Route inputs to nodes
        node_inputs = []
        for node_idx in range(self.output_size):
            # Get input indices for this node
            indices = self.routing[node_idx]
            
            # Extract routed input
            routed = x[:, indices]  # (batch, n)
            
            # Apply bit flipping augmentation if in training
            if self.training and self.layer_config:
                flip_prob = self.layer_config.flip_probability
                if flip_prob > 0:
                    flips = torch.bernoulli(
                        torch.ones_like(routed) * flip_prob
                    )
                    routed = routed ^ flips.bool()  # XOR flip
            
            node_inputs.append(routed)
        
        # Compute node outputs
        outputs = []
        for node_idx, node_input in enumerate(node_inputs):
            node = self.nodes[node_idx]
            node_output = node(node_input)
            outputs.append(node_output)
        
        # Concatenate outputs
        output = torch.cat(outputs, dim=1)  # (batch, output_size)
        
        # Apply gradient stabilization if configured
        if self.training and self.layer_config:
            output = self._apply_gradient_stabilization(output)
        
        return output
    
    def _apply_gradient_stabilization(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient stabilization during backward pass."""
        if self.layer_config.grad_stabilization == 'none':
            return x
        
        # Gradient stabilization hook
        if not hasattr(self, '_grad_stabilizer_registered'):
            def stabilize_gradients(grad):
                if self.layer_config.grad_stabilization == 'layerwise':
                    # Normalize across all elements
                    std = torch.std(grad)
                    target_std = self.layer_config.grad_target_std
                    if std > self.layer_config.grad_epsilon:
                        grad = grad * (target_std / (std + self.layer_config.grad_epsilon))
                
                elif self.layer_config.grad_stabilization == 'nodewise':
                    # Normalize per node
                    # Reshape to (batch, num_nodes)
                    batch_size = grad.shape[0]
                    grad_reshaped = grad.reshape(batch_size, self.output_size)
                    
                    # Normalize each node's gradients
                    for node_idx in range(self.output_size):
                        node_grad = grad_reshaped[:, node_idx]
                        std = torch.std(node_grad)
                        target_std = self.layer_config.grad_target_std
                        if std > self.layer_config.grad_epsilon:
                            node_grad = node_grad * (target_std / (std + self.layer_config.grad_epsilon))
                            grad_reshaped[:, node_idx] = node_grad
                    
                    grad = grad_reshaped.reshape_as(grad)
                
                return grad
            
            x.register_hook(stabilize_gradients)
            self._grad_stabilizer_registered = True
        
        return x
```

### Step 3: Use the Layer

```python
from difflut.layers.layer_config import LayerConfig

# Create layer configuration with training augmentation
layer_config = LayerConfig(
    flip_probability=0.05,
    grad_stabilization='layerwise',
    grad_target_std=1.0,
)

# Create layer
layer = RandomRoutingLayer(
    input_size=512,
    output_size=256,
    node_type='probabilistic',
    n=6,
    layer_config=layer_config,
    seed=42
)

# Use in forward pass
x = torch.randn(32, 512)  # (batch=32, features=512)
y = layer(x)  # (batch=32, output_size=256)
```

---

## Advanced Examples

### Example 1: Learnable Routing Layer

```python
@register_layer('learnable_routing')
class LearnableRoutingLayer(BaseLUTLayer):
    """Layer with learned routing weights."""
    
    def __init__(
        self,
        input_size: int = DEFAULT_INPUT_SIZE,
        output_size: int = DEFAULT_OUTPUT_SIZE,
        node_type: str = DEFAULT_NODE_TYPE,
        n: int = DEFAULT_INPUT_DIM,
        node_kwargs: Optional[Dict[str, Any]] = None,
        layer_config: Optional[LayerConfig] = None,
        **kwargs
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            node_type=node_type,
            n=n,
            node_kwargs=node_kwargs,
            layer_config=layer_config,
            **kwargs
        )
        
        # Learnable routing matrix
        self.routing_weights = nn.Parameter(
            torch.randn(output_size, input_size) / (input_size ** 0.5)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with learned soft routing."""
        # x: (batch, input_size)
        batch_size = x.shape[0]
        
        outputs = []
        for node_idx in range(self.output_size):
            # Compute soft routing weights for this node
            weights = torch.softmax(self.routing_weights[node_idx], dim=0)
            
            # Weighted sum of inputs
            node_input = torch.mv(weights.unsqueeze(0).expand(batch_size, -1), x.t()).t()
            # This is actually: sum_over_features(x * weights)
            # Better approach:
            node_input = torch.matmul(x, weights.unsqueeze(1)).squeeze(1)  # (batch,)
            
            # But we need n-dimensional input per node...
            # So select top-n features by attention
            top_k_indices = torch.topk(weights, n=self.n, largest=True)[1]
            routed = x[:, top_k_indices]  # (batch, n)
            
            # Apply node
            node = self.nodes[node_idx]
            node_output = node(routed)
            outputs.append(node_output)
        
        output = torch.cat(outputs, dim=1)
        
        if self.training and self.layer_config:
            output = self._apply_gradient_stabilization(output)
        
        return output
    
    def _apply_gradient_stabilization(self, x: torch.Tensor) -> torch.Tensor:
        """Gradient stabilization similar to RandomRoutingLayer."""
        if self.layer_config.grad_stabilization == 'none':
            return x
        
        # Implementation as in RandomRoutingLayer
        return x
```

### Example 2: Feature-Wise Grouping Layer

```python
@register_layer('feature_wise_groups')
class FeatureWiseGroupingLayer(BaseLUTLayer):
    """Group features for improved interpretability."""
    
    def __init__(
        self,
        input_size: int = DEFAULT_INPUT_SIZE,
        output_size: int = DEFAULT_OUTPUT_SIZE,
        node_type: str = DEFAULT_NODE_TYPE,
        n: int = DEFAULT_INPUT_DIM,
        node_kwargs: Optional[Dict[str, Any]] = None,
        layer_config: Optional[LayerConfig] = None,
        num_groups: Optional[int] = None,
        **kwargs
    ):
        # Auto-determine groups if not specified
        if num_groups is None:
            num_groups = max(1, input_size // n)
        
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            node_type=node_type,
            n=n,
            node_kwargs=node_kwargs,
            layer_config=layer_config,
            **kwargs
        )
        
        self.num_groups = num_groups
        self.group_size = input_size // num_groups
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with grouped feature routing."""
        batch_size = x.shape[0]
        outputs = []
        
        for node_idx in range(self.output_size):
            # Select group for this node (round-robin)
            group_idx = node_idx % self.num_groups
            start = group_idx * self.group_size
            end = start + min(self.group_size, self.n)
            
            # Extract group
            routed = x[:, start:end]  # (batch, <= n)
            
            # Pad if necessary
            if routed.shape[1] < self.n:
                padding = torch.zeros(
                    batch_size, self.n - routed.shape[1],
                    device=x.device
                )
                routed = torch.cat([routed, padding], dim=1)
            
            # Apply node
            node = self.nodes[node_idx]
            node_output = node(routed)
            outputs.append(node_output)
        
        output = torch.cat(outputs, dim=1)
        
        if self.training and self.layer_config:
            output = self._apply_gradient_stabilization(output)
        
        return output
    
    def _apply_gradient_stabilization(self, x: torch.Tensor) -> torch.Tensor:
        """Gradient stabilization."""
        if self.layer_config.grad_stabilization == 'none':
            return x
        return x
```

---

## Testing Your Layer

Create `tests/test_layers/test_my_layer.py`:

```python
import torch
import pytest
from difflut.layers import MyCustomLayer
from difflut.layers.layer_config import LayerConfig
from difflut.registry import REGISTRY

class TestMyCustomLayer:
    
    def test_registration(self):
        """Verify layer is registered."""
        layer_class = REGISTRY.get_layer('my_custom_layer')
        assert layer_class is not None
    
    def test_initialization(self):
        """Test layer initialization."""
        layer = MyCustomLayer(
            input_size=512,
            output_size=256,
            node_type='probabilistic',
            n=6
        )
        assert layer.input_size == 512
        assert layer.output_size == 256
    
    def test_forward_shape(self):
        """Test forward output shape."""
        layer = MyCustomLayer(
            input_size=512,
            output_size=256,
            node_type='probabilistic',
            n=6
        )
        x = torch.randn(32, 512)
        y = layer(x)
        
        assert y.shape == (32, 256)
    
    def test_with_layer_config(self):
        """Test layer with training configuration."""
        layer_config = LayerConfig(
            flip_probability=0.05,
            grad_stabilization='layerwise'
        )
        
        layer = MyCustomLayer(
            input_size=512,
            output_size=256,
            node_type='probabilistic',
            n=6,
            layer_config=layer_config
        )
        
        # Training mode - should apply augmentation
        layer.train()
        x = torch.randn(32, 512)
        y_train = layer(x)
        
        # Eval mode - no augmentation
        layer.eval()
        y_eval = layer(x)
        
        # Outputs may differ due to stochastic augmentation
        assert y_train.shape == (32, 256)
        assert y_eval.shape == (32, 256)
    
    def test_gradient_flow(self):
        """Test that gradients flow through layer."""
        layer = MyCustomLayer(
            input_size=512,
            output_size=256,
            node_type='probabilistic',
            n=6
        )
        
        x = torch.randn(32, 512, requires_grad=True)
        y = layer(x)
        
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_deterministic_eval(self):
        """Test eval mode is deterministic."""
        layer = MyCustomLayer(
            input_size=512,
            output_size=256,
            node_type='probabilistic',
            n=6
        )
        layer.eval()
        
        x = torch.randn(32, 512)
        y1 = layer(x)
        y2 = layer(x)
        
        assert torch.allclose(y1, y2)
    
    def test_device_transfer(self, device):
        """Test layer works after device transfer."""
        layer = MyCustomLayer(
            input_size=512,
            output_size=256,
            node_type='probabilistic',
            n=6
        ).to(device)
        
        x = torch.randn(32, 512, device=device)
        y = layer(x)
        
        assert y.device == device
```

---

## Key Patterns

1. **Module-Level Defaults**: Use CAPITALS for constants
2. **LayerConfig**: Use for training parameters (not structural)
3. **Type Hints**: Full PEP 484 type hints
4. **Training Augmentation**: Bit flipping, gradient stabilization
5. **Routing**: How inputs map to nodes (random, learnable, grouped)
6. **Node Management**: BaseLUTLayer creates and manages nodes
7. **Gradient Stabilization**: Apply during backward via hooks
8. **Docstrings**: NumPy format with all parameters

---

## Next Steps

1. **Review existing layers** in `difflut/layers/` for patterns
2. **Design routing connectivity** (random, learnable, feature-wise)
3. **Implement forward()** with routing logic
4. **Add training augmentation** via LayerConfig
5. **Add gradient stabilization** if needed
6. **Add comprehensive tests** following test patterns
7. **Integrate with pipelines** via REGISTRY

---

## Resources

- **Base Class**: `difflut/layers/base_layer.py`
- **Config Class**: `difflut/layers/layer_config.py`
- **Examples**: `difflut/layers/random_layer.py`, `difflut/layers/learnable_layer.py`
- **Tests**: `tests/test_layers/`
- **User Guide**: [Layers Guide](../../USER_GUIDE/components/layers.md)
- **Registry**: [Registry & Pipeline Guide](../../USER_GUIDE/registry_pipeline.md)
