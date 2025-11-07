# Creating Custom Components

Learn how to implement and register custom DiffLUT components (nodes, encoders, layers).

## Overview

All DiffLUT components follow an extensible design:

1. **Extend** the appropriate base class (`BaseNode`, `BaseEncoder`, `BaseLUTLayer`)
2. **Register** using the decorator system (`@register_node`, etc.)
3. **Implement** required methods for your component
4. **Test** thoroughly
5. **Optionally add** CUDA support for performance

### Type-Safe Configuration

DiffLUT uses type-safe configuration classes for nodes and layers:

- **NodeConfig**: Type-safe parameters for nodes (input/output dimensions, initializers, regularizers, node-specific params)
- **LayerConfig**: Type-safe parameters for layer training (bit flipping, gradient stabilization)

```python
from difflut.nodes.node_config import NodeConfig
from difflut.layers.layer_config import LayerConfig

# Node configuration
node_config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=kaiming_normal_init,
    init_kwargs={'a': 0.0, 'mode': 'fan_in'},
    regularizers={'l2': l2_regularizer},
    extra_params={'use_cuda': True}  # Node-specific parameters
)

# Layer training configuration
layer_config = LayerConfig(
    flip_probability=0.1,
    grad_stabilization='layerwise',
    grad_target_std=1.0
)
```

**Why use these?**
- Type safety and IDE autocomplete
- Consistent parameter validation
- Clear separation of node logic vs. layer training augmentation
- Easy serialization for configuration files
- Better error messages

When creating custom components, ensure they work with these configuration classes:

```python
# Your custom node should accept NodeConfig parameters
class MyCustomNode(BaseNode):
    def __init__(self, input_dim, output_dim, init_fn=None, init_kwargs=None, 
                 regularizers=None, my_custom_param=None, **kwargs):
        super().__init__(input_dim, output_dim, init_fn, init_kwargs, regularizers, **kwargs)
        # Custom logic with my_custom_param (passed via extra_params)
        self.my_custom_param = kwargs.get('my_custom_param', my_custom_param)

# Your custom layer should accept LayerConfig
class MyCustomLayer(BaseLUTLayer):
    def __init__(self, input_size, output_size, node_type, n, 
                 node_kwargs=None, layer_config=None, **kwargs):
        super().__init__(input_size, output_size, node_type, n, node_kwargs, layer_config, **kwargs)
        # layer_config contains flip_probability, grad_stabilization, etc.
```

---

## Custom Nodes

### Creating a Custom Node

Nodes define computation at individual LUT units. Extend `BaseNode`:

```python
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
from difflut.nodes import BaseNode
from difflut.nodes.node_config import NodeConfig
from difflut.utils.warnings import warn_default_value
from difflut import register_node

# Define module defaults at the top
DEFAULT_MY_NODE_INPUT_DIM: int = 6
DEFAULT_MY_NODE_OUTPUT_DIM: int = 1

@register_node('my_custom_node')
class MyCustomNode(BaseNode):
    """
    My custom LUT node implementation.
    
    This node demonstrates the recommended pattern for all new components.
    Processes 2D tensors: (batch_size, input_dim) → (batch_size, output_dim)
    """
    
    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        init_fn: Optional[Callable[[torch.Tensor], None]] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        regularizers: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Initialize the custom node.
        
        Parameters
        ----------
        input_dim : Optional[int]
            Number of binary inputs. Defaults to DEFAULT_MY_NODE_INPUT_DIM.
        output_dim : Optional[int]
            Number of outputs. Defaults to DEFAULT_MY_NODE_OUTPUT_DIM.
        init_fn : Optional[Callable]
            Initialization function for weights.
        init_kwargs : Optional[Dict[str, Any]]
            Arguments for initialization function.
        regularizers : Optional[Dict[str, Any]]
            Custom regularization functions.
        **kwargs : dict
            Additional arguments (for compatibility).
        
        Note
        ----
        Each node instance processes 2D tensors independently.
        Layers create multiple node instances using nn.ModuleList.
        """
        # Apply defaults with warnings
        if input_dim is None:
            input_dim = DEFAULT_MY_NODE_INPUT_DIM
            warn_default_value("input_dim", input_dim, stacklevel=2)
        
        if output_dim is None:
            output_dim = DEFAULT_MY_NODE_OUTPUT_DIM
            warn_default_value("output_dim", output_dim, stacklevel=2)
        
        # Call parent constructor (no layer_size parameter)
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            init_fn=init_fn,
            init_kwargs=init_kwargs,
            regularizers=regularizers,
            **kwargs
        )
        
        # Initialize parameters: shape (2^input_dim, output_dim)
        num_lut_entries = 2 ** self.num_inputs
        
        self.weights = nn.Parameter(
            torch.randn(num_lut_entries, self.num_outputs) * 0.01
        )
    
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training.
        
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch_size, input_dim) with binary values {0, 1}
        
        Returns
        -------
        torch.Tensor
            Output of shape (batch_size, output_dim)
        
        Note
        ----
        Each node instance processes 2D tensors independently.
        No layer_size dimension - layers manage multiple nodes via nn.ModuleList.
        """
        # Convert binary input to table indices
        indices = self._binary_to_index(x)
        
        # Look up and return weights
        return self.weights[indices]
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during evaluation.
        
        Can differ from training (e.g., quantized version).
        Default: same as training.
        
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, num_inputs) with binary values
        
        Returns
        -------
        torch.Tensor
            Output of shape (batch, num_outputs)
        """
        return self.forward_train(x)
    
    def regularization(self) -> torch.Tensor:
        """
        Compute optional regularization loss.
        
        Returns
        -------
        torch.Tensor
            Scalar tensor representing regularization term
        """
        # Example: L2 regularization on weights
        return 0.001 * torch.sum(self.weights ** 2)
    
    def export_bitstream(self) -> Any:
        """
        Export LUT configuration for FPGA deployment.
        
        Returns
        -------
        numpy.ndarray
            Array of shape (2^num_inputs, num_outputs)
        """
        return self.weights.detach().cpu().numpy()
```

**Key pattern elements:**
1. **PEP 484 types** - Use `Optional[T]` instead of `T | None`
2. **Module constants** - Define defaults at module level in CAPITALS
3. **Default warnings** - Use `warn_default_value()` for traceability
4. **Type-safe** - Full type hints enable IDE support and mypy checking
5. **NumPy docstrings** - Document all parameters and return values

### Base Class Methods

The `BaseNode` class provides these helper methods:

```python
class MyNode(BaseNode):
    def forward_train(self, x):
        # Convert binary indices to table indices
        # x: (batch, num_inputs) binary values
        # returns: (batch,) indices into weight table
        indices = self._binary_to_index(x)
        return self.weights[indices]
```

### Complete Node Example: Polynomial LUT

```python
@register_node('poly_example')
class PolyExampleNode(BaseNode):
    """
    Polynomial approximation LUT node.
    Processes 2D tensors: (batch_size, input_dim) → (batch_size, output_dim)
    """
    
    def __init__(self, input_dim=None, output_dim=None, degree=3, **kwargs):
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)
        self.degree = degree
        
        # Store polynomial coefficients for each output
        self.coeffs = nn.ParameterList([
            nn.Parameter(torch.randn(degree + 1))
            for _ in range(self.num_outputs)
        ])
    
    def forward_train(self, x):
        """
        Forward pass with 2D tensors.
        
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, input_dim) with binary values
        
        Returns
        -------
        torch.Tensor
            Shape (batch_size, output_dim)
        """
        # Treat binary input as continuous value in [0, 1]
        x_continuous = x.float().mean(dim=1, keepdim=True)
        
        # Evaluate polynomial
        output = torch.zeros(x.size(0), self.num_outputs, device=x.device)
        for i, coeff in enumerate(self.coeffs):
            # Horner's method for polynomial evaluation
            poly_val = coeff[-1]
            for j in range(len(coeff) - 2, -1, -1):
                poly_val = poly_val * x_continuous + coeff[j]
            output[:, i] = poly_val.squeeze(-1)
        
        return output
    
    def forward_eval(self, x):
        return self.forward_train(x)
    
    def regularization(self):
        total = 0
        for coeff in self.coeffs:
            total += torch.sum(coeff ** 2)
        return 0.001 * total
    
    def export_bitstream(self):
        # Convert polynomial to LUT values
        lut = torch.zeros(2 ** self.num_inputs, self.num_outputs)
        for i in range(2 ** self.num_inputs):
            x_normalized = i / (2 ** self.num_inputs - 1)
            for j, coeff in enumerate(self.coeffs):
                poly_val = coeff[-1]
                for k in range(len(coeff) - 2, -1, -1):
                    poly_val = poly_val * x_normalized + coeff[k]
                lut[i, j] = poly_val
        return lut.cpu().numpy()
```

## Custom Encoders

### Creating a Custom Encoder

Encoders transform continuous inputs to discrete representations. Extend `BaseEncoder`:

```python
import torch
import torch.nn as nn
from typing import Optional
from difflut.encoder import BaseEncoder
from difflut.utils.warnings import warn_default_value
from difflut import register_encoder

# Module defaults
DEFAULT_ENCODER_NUM_BITS: int = 8
DEFAULT_ENCODER_FEATURE_WISE: bool = True

@register_encoder('my_custom_encoder')
class MyCustomEncoder(BaseEncoder):
    """
    My custom input encoder.
    
    Transforms continuous values to discrete binary representations.
    """
    
    def __init__(
        self,
        num_bits: Optional[int] = None,
        feature_wise: bool = DEFAULT_ENCODER_FEATURE_WISE,
        flatten: bool = True
    ) -> None:
        """
        Initialize the custom encoder.
        
        Parameters
        ----------
        num_bits : Optional[int]
            Number of bits per feature. 
            Defaults to DEFAULT_ENCODER_NUM_BITS.
        feature_wise : bool
            If True, fit each feature independently.
            Default is True.
        flatten : bool
            If True, return 2D tensor. If False, return 3D.
            Default is True.
        """
        super().__init__()
        
        # Apply defaults with warnings
        if num_bits is None:
            num_bits = DEFAULT_ENCODER_NUM_BITS
            warn_default_value("num_bits", num_bits, stacklevel=2)
        
        self.num_bits = num_bits
        self.feature_wise = feature_wise
        self.flatten = flatten
        
        # Register buffers for fitted statistics
        self.register_buffer('min_vals', None)
        self.register_buffer('max_vals', None)
        self._is_fitted = False
    
    def fit(self, data: torch.Tensor) -> 'MyCustomEncoder':
        """
        Fit encoder to data (learn statistics).
        
        Parameters
        ----------
        data : torch.Tensor
            Training data of shape (N, D) with continuous values
        
        Returns
        -------
        MyCustomEncoder
            Returns self for method chaining
        """
        if self.feature_wise:
            # Fit per feature
            self.min_vals = torch.min(data, dim=0)[0]
            self.max_vals = torch.max(data, dim=0)[0]
        else:
            # Fit globally
            min_val = torch.min(data)
            max_val = torch.max(data)
            self.min_vals = torch.full(data.shape[1:], min_val)
            self.max_vals = torch.full(data.shape[1:], max_val)
        
        self._is_fitted = True
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous input to discrete representation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (N, D) with continuous values
        
        Returns
        -------
        torch.Tensor
            Encoded data. Shape:
            - (N, D * num_bits) if flatten=True
            - (N, D, num_bits) if flatten=False
        """
        """
        if not self._is_fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        
        # Normalize to [0, 1]
        x_normalized = (x - self.min_vals) / (self.max_vals - self.min_vals + 1e-8)
        x_normalized = torch.clamp(x_normalized, 0, 1)
        
        # Encode to bits
        # Your custom encoding logic here
        # Example: simple binary encoding
        encoded_list = []
        for bit in range(self.num_bits):
            threshold = (2 ** (bit + 1) - 1) / (2 ** self.num_bits)
            encoded_list.append((x_normalized > threshold).float())
        
        encoded = torch.cat(encoded_list, dim=1)
        return encoded
```

### Complete Encoder Example: Modified Gray Code

```python
@register_encoder('custom_gray')
class CustomGrayEncoder(BaseEncoder):
    """Custom Gray code encoder with adaptive resolution."""
    
    def __init__(self, num_bits=8, feature_wise=True, adapt_resolution=True):
        super().__init__()
        self.num_bits = num_bits
        self.feature_wise = feature_wise
        self.adapt_resolution = adapt_resolution
        
        self.register_buffer('min_vals', None)
        self.register_buffer('max_vals', None)
        self.register_buffer('quantiles', None)
        self._is_fitted = False
    
    def fit(self, data):
        """Fit encoder learning quantiles."""
        if self.feature_wise:
            self.min_vals = torch.min(data, dim=0)[0]
            self.max_vals = torch.max(data, dim=0)[0]
            
            if self.adapt_resolution:
                # Learn quantiles for better resolution
                quantiles = []
                for i in range(data.shape[1]):
                    q = torch.quantile(
                        data[:, i],
                        torch.linspace(0, 1, 2 ** self.num_bits)
                    )
                    quantiles.append(q)
                self.quantiles = torch.stack(quantiles)
        else:
            self.min_vals = torch.full(data.shape[1:], torch.min(data))
            self.max_vals = torch.full(data.shape[1:], torch.max(data))
        
        self._is_fitted = True
    
    def _binary_to_gray(self, binary):
        """Convert binary to Gray code."""
        gray = binary.clone()
        for i in range(1, binary.shape[-1]):
            gray[..., i] = binary[..., i] ^ binary[..., i-1]
        return gray
    
    def forward(self, x):
        """Encode to Gray code."""
        if not self._is_fitted:
            raise RuntimeError("Encoder not fitted.")
        
        # Quantize input
        if self.adapt_resolution and self.quantiles is not None:
            # Use learned quantiles
            x_normalized = torch.zeros_like(x)
            for i in range(x.shape[1]):
                # Find nearest quantile
                diffs = torch.abs(x[:, i:i+1] - self.quantiles[i])
                indices = torch.argmin(diffs, dim=1)
                x_normalized[:, i] = indices / (2 ** self.num_bits - 1)
        else:
            x_normalized = (x - self.min_vals) / (self.max_vals - self.min_vals + 1e-8)
        
        # Convert to integer indices
        indices = (x_normalized * (2 ** self.num_bits - 1)).long()
        
        # Convert to binary
        binary = torch.zeros(
            (*indices.shape, self.num_bits),
            device=x.device,
            dtype=torch.float
        )
        for i in range(self.num_bits):
            binary[..., i] = ((indices >> i) & 1).float()
        
        # Convert to Gray code
        gray = self._binary_to_gray(binary)
        
        # Flatten to (N, D * num_bits)
        return gray.reshape(x.shape[0], -1)
```

## Custom Layers

### Creating a Custom Layer

Layers define connectivity between inputs and LUT nodes. Extend `BaseLUTLayer`:

```python
import torch
import torch.nn as nn
from difflut.layers import BaseLUTLayer
from difflut.nodes.node_config import NodeConfig
from difflut.layers.layer_config import LayerConfig
from difflut import register_layer

@register_layer('my_custom_layer')
class MyCustomLayer(BaseLUTLayer):
    """My custom layer connectivity."""
    
    def __init__(self, input_size, output_size, node_type, n, 
                 node_kwargs=None, layer_config=None, custom_param=None):
        """
        Initialize layer.
        
        Args:
            input_size: Number of input features
            output_size: Number of output LUT nodes
            node_type: Node class to instantiate
            n: Number of inputs per LUT node
            node_kwargs: NodeConfig instance or dict of parameters for nodes
            layer_config: LayerConfig instance for training parameters (optional)
            custom_param: Your custom parameter
        """
        super().__init__(input_size, output_size, node_type, n, node_kwargs, layer_config)
        self.custom_param = custom_param
        
        # Define connectivity pattern
        # Example: create custom routing matrix
        self.routing_matrix = nn.Parameter(
            torch.randn(output_size, input_size) / (input_size ** 0.5)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, input_size)
        
        Returns:
            Output tensor (batch, output_size)
        """
        # Route inputs to nodes based on your connectivity pattern
        # Example: soft routing
        routing_weights = torch.softmax(self.routing_matrix, dim=1)
        
        # Apply routing (simplified example)
        routed = torch.matmul(x, routing_weights.t())
        
        # Binarize for LUT nodes
        binary_input = (routed > 0).float()
        
        # Process through nodes
        output = torch.zeros(x.size(0), self.output_size, device=x.device)
        for i, node in enumerate(self.nodes):
            # Get inputs for this node
            node_input = binary_input[:, i:i+self.n]
            output[:, i] = node(node_input).squeeze(-1)
        
        return output
```

### Complete Layer Example: Feature-Wise Grouping

```python
@register_layer('feature_wise')
class FeatureWiseGroupingLayer(BaseLUTLayer):
    """Group features for improved interpretability."""
    
    def __init__(self, input_size, output_size, node_type, n,
                 num_groups=None, node_kwargs=None, layer_config=None):
        super().__init__(input_size, output_size, node_type, n, node_kwargs, layer_config)
        
        if num_groups is None:
            num_groups = min(8, output_size)
        self.num_groups = num_groups
        
        # Assign inputs to groups
        self.group_assignment = nn.Parameter(
            torch.randn(input_size, num_groups),
            requires_grad=False
        )
    
    def forward(self, x):
        """Route based on feature groups."""
        # Get group assignments
        group_weights = torch.softmax(self.group_assignment, dim=1)
        
        # Assign features to groups
        feature_groups = torch.matmul(x, group_weights)
        
        # Process each group
        output = []
        group_idx = 0
        for node in self.nodes:
            if group_idx >= len(self.nodes):
                group_idx = 0
            
            # Get inputs from appropriate group
            group_features = feature_groups[:, group_idx % self.num_groups]
            
            # Binarize
            binary_input = (group_features > 0).float()
            
            # Process through node
            node_output = node(binary_input)
            output.append(node_output)
            
            group_idx += 1
        
        return torch.cat(output, dim=1)
```

## CUDA Support

### Adding CUDA Kernels

For compute-intensive nodes, add CUDA support:

1. **Create CUDA kernel file** `difflut/nodes/cuda/my_kernel.cu`
2. **Create Python wrapper** `difflut/nodes/cuda/my_kernel.py`
3. **Update setup.py** to compile kernel
4. **Add CPU fallback** for when CUDA unavailable

Example CUDA kernel setup:

```python
# difflut/nodes/cuda/my_kernel.py

try:
    import torch
    from torch.utils.cpp_extension import load
    
    # Load CUDA extension
    my_kernel = load(
        'my_kernel',
        sources=['difflut/nodes/cuda/my_kernel.cu'],
        verbose=False,
    )
except Exception:
    my_kernel = None

def apply_my_kernel(x, weights):
    """Apply custom CUDA kernel with fallback."""
    if my_kernel is not None and x.is_cuda:
        return my_kernel.forward(x, weights)
    else:
        # CPU fallback
        return torch.tensordot(x, weights, dims=[[1], [0]])
```

### Node with CUDA Support

```python
@register_node('cuda_node')
class CUDANode(BaseNode):
    """
    CUDA-accelerated node with CPU fallback.
    Processes 2D tensors: (batch_size, input_dim) → (batch_size, output_dim)
    """
    def __init__(self, input_dim=None, output_dim=None, **kwargs):
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)
        # Weights shape: (2^input_dim, output_dim)
        self.weights = nn.Parameter(
            torch.randn(2 ** self.num_inputs, self.num_outputs)
        )
    
    def forward_train(self, x):
        if x.is_cuda:
            try:
                from difflut.nodes.cuda import apply_my_kernel
                return apply_my_kernel(x.float(), self.weights)
            except Exception:
                pass
        
        # CPU fallback
        indices = self._binary_to_index(x)
        return self.weights[indices]
    
    def forward_eval(self, x):
        return self.forward_train(x)
```

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

## Registering Your Component

Once implemented, components are automatically registered and can be accessed via the REGISTRY:

```python
from difflut.registry import REGISTRY
from difflut.nodes.node_config import NodeConfig

# Get your registered component
NodeClass = REGISTRY.get_node('my_custom_node')

# Create instance with type-safe configuration
node_config = NodeConfig(input_dim=4, output_dim=1)
node = NodeClass(**node_config.to_dict())

# Or instantiate directly with kwargs
node = NodeClass(input_dim=4, output_dim=1)
```

### Verifying Registration

```python
from difflut.registry import REGISTRY

# Check if your component is registered
if 'my_custom_node' in REGISTRY.list_nodes():
    print("✓ Component successfully registered")

# List all registered components
print("Available nodes:", REGISTRY.list_nodes())
print("Available encoders:", REGISTRY.list_encoders())
print("Available layers:", REGISTRY.list_layers())
```

### Using Custom Components in Pipelines

Once registered, your custom components work seamlessly with DiffLUT's configuration-driven pipeline system:

```python
from difflut.registry import REGISTRY
from difflut.nodes.node_config import NodeConfig
from difflut.layers.layer_config import LayerConfig

# Build layer with your custom node
MyNodeClass = REGISTRY.get_node('my_custom_node')
LayerClass = REGISTRY.get_layer('random')

node_config = NodeConfig(
    input_dim=6,
    output_dim=1,
    extra_params={'my_custom_param': 42}  # Your custom parameters
)

layer_config = LayerConfig(
    flip_probability=0.1,
    grad_stabilization='layerwise'
)

layer = LayerClass(
    input_size=512,
    output_size=256,
    node_type=MyNodeClass,
    n=6,
    node_kwargs=node_config,
    layer_config=layer_config
)
```

### Configuration File Integration

Your custom components can be used in YAML/JSON configuration files:

```yaml
# model_config.yaml
layers:
  - name: custom_layer
    type: random
    node_type: my_custom_node  # Your registered name
    input_size: 512
    output_size: 256
    n: 6
    node_params:
      input_dim: 6
      output_dim: 1
      my_custom_param: 42  # Goes into extra_params
    layer_training:
      flip_probability: 0.1
      grad_stabilization: layerwise
```

Then load and build:

```python
import yaml
from difflut.registry import REGISTRY
from difflut.nodes.node_config import NodeConfig
from difflut.layers.layer_config import LayerConfig

with open('model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

layer_cfg = config['layers'][0]
NodeClass = REGISTRY.get_node(layer_cfg['node_type'])
LayerClass = REGISTRY.get_layer(layer_cfg['type'])

# Build NodeConfig
node_params = layer_cfg['node_params']
extra_params = {k: v for k, v in node_params.items() 
                if k not in ['input_dim', 'output_dim']}

node_config = NodeConfig(
    input_dim=node_params['input_dim'],
    output_dim=node_params['output_dim'],
    extra_params=extra_params
)

# Build LayerConfig
layer_config = LayerConfig(**layer_cfg.get('layer_training', {}))

# Create layer
layer = LayerClass(
    input_size=layer_cfg['input_size'],
    output_size=layer_cfg['output_size'],
    node_type=NodeClass,
    n=layer_cfg['n'],
    node_kwargs=node_config,
    layer_config=layer_config
)
```

For more details on configuration-driven pipelines, see the [Registry & Pipeline Guide](../USER_GUIDE/registry_pipeline.md).

---

## Next Steps

1. **Review existing implementations** in `difflut/` for patterns
2. **Start simple** - begin with a basic component
3. **Add tests** - comprehensive test coverage
4. **Consider CUDA** - for performance-critical nodes
5. **Use in pipelines** - see [Registry & Pipeline Guide](../USER_GUIDE/registry_pipeline.md) for integration
6. **Submit PR** - follow [Contributing Guide](contributing.md)

## Resources

- **Base Classes**: `difflut/nodes/base_node.py`, `difflut/encoder/base_encoder.py`, `difflut/layers/base_layer.py`
- **Configuration Classes**: `difflut/nodes/node_config.py`, `difflut/layers/layer_config.py`
- **Registry System**: `difflut/registry.py`
- **Examples**: Existing implementations in respective directories
- **Tests**: `tests/` directory
- **User Guides**: [Components Guide](../USER_GUIDE/components.md), [Registry & Pipeline Guide](../USER_GUIDE/registry_pipeline.md)
