# Creating Custom Components

Learn how to implement and register custom DiffLUT components (nodes, encoders, layers).

## Overview

All DiffLUT components follow an extensible design:

1. **Extend** the appropriate base class (`BaseNode`, `BaseEncoder`, `BaseLUTLayer`)
2. **Register** using the decorator system (`@register_node`, etc.)
3. **Implement** required methods for your component
4. **Test** thoroughly
5. **Optionally add** CUDA support for performance

## Custom Nodes

### Creating a Custom Node

Nodes define computation at individual LUT units. Extend `BaseNode`:

```python
import torch
import torch.nn as nn
from difflut.nodes import BaseNode
from difflut import register_node

@register_node('my_custom_node')
class MyCustomNode(BaseNode):
    """My custom LUT node implementation."""
    
    def __init__(self, input_dim=None, output_dim=None, **kwargs):
        """
        Initialize the node.
        
        Args:
            input_dim: List [n] - number of binary inputs
            output_dim: List [m] - number of outputs
            **kwargs: Additional arguments
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)
        
        # Initialize parameters
        # self.num_inputs and self.num_outputs are set by parent class
        num_lut_entries = 2 ** self.num_inputs
        
        self.weights = nn.Parameter(
            torch.randn(num_lut_entries, self.num_outputs) * 0.01
        )
    
    def forward_train(self, x):
        """
        Forward pass during training.
        
        Args:
            x: Tensor of shape (batch, num_inputs) with binary values {0, 1}
        
        Returns:
            Tensor of shape (batch, num_outputs)
        """
        # Convert binary input to table indices
        indices = self._binary_to_index(x)
        
        # Look up and return weights
        return self.weights[indices]
    
    def forward_eval(self, x):
        """
        Forward pass during evaluation.
        
        Can be different from training (e.g., quantized version).
        Default: same as training.
        
        Args:
            x: Tensor of shape (batch, num_inputs) with binary values
        
        Returns:
            Tensor of shape (batch, num_outputs)
        """
        return self.forward_train(x)
    
    def regularization(self):
        """
        Optional regularization loss.
        
        Returns:
            Scalar tensor representing regularization term
        """
        # Example: L2 regularization on weights
        return 0.001 * torch.sum(self.weights ** 2)
    
    def export_bitstream(self):
        """
        Export LUT configuration for FPGA.
        
        Returns:
            numpy array of shape (2^num_inputs, num_outputs)
        """
        return self.weights.detach().cpu().numpy()
```

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
    """Polynomial approximation LUT node."""
    
    def __init__(self, input_dim=None, output_dim=None, degree=3, **kwargs):
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)
        self.degree = degree
        
        # Store polynomial coefficients for each output
        self.coeffs = nn.ParameterList([
            nn.Parameter(torch.randn(degree + 1))
            for _ in range(self.num_outputs)
        ])
    
    def forward_train(self, x):
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
from difflut.encoder import BaseEncoder
from difflut import register_encoder

@register_encoder('my_custom_encoder')
class MyCustomEncoder(BaseEncoder):
    """My custom input encoder."""
    
    def __init__(self, num_bits=8, feature_wise=True):
        """
        Initialize encoder.
        
        Args:
            num_bits: Number of bits per feature
            feature_wise: If True, fit each feature independently
        """
        super().__init__()
        self.num_bits = num_bits
        self.feature_wise = feature_wise
        
        # Register buffers for fitted statistics
        self.register_buffer('min_vals', None)
        self.register_buffer('max_vals', None)
        self._is_fitted = False
    
    def fit(self, data):
        """
        Fit encoder to data (learn statistics).
        
        Args:
            data: Tensor of shape (N, D) with continuous values
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
    
    def forward(self, x):
        """
        Encode continuous input to discrete representation.
        
        Args:
            x: Tensor of shape (N, D) with continuous values
        
        Returns:
            Tensor of shape (N, D * num_bits) with encoded values
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
from difflut import register_layer

@register_layer('my_custom_layer')
class MyCustomLayer(BaseLUTLayer):
    """My custom layer connectivity."""
    
    def __init__(self, input_size, output_size, node_type, n, 
                 node_kwargs=None, custom_param=None):
        """
        Initialize layer.
        
        Args:
            input_size: Number of input features
            output_size: Number of output LUT nodes
            node_type: Node class to instantiate
            n: Number of inputs per LUT node
            node_kwargs: Dict of parameters for nodes
            custom_param: Your custom parameter
        """
        super().__init__(input_size, output_size, node_type, n, node_kwargs)
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
                 num_groups=None, node_kwargs=None):
        super().__init__(input_size, output_size, node_type, n, node_kwargs)
        
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
    def __init__(self, input_dim=None, output_dim=None, **kwargs):
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)
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

### Writing Tests

```python
import pytest
import torch
from difflut.nodes import MyCustomNode
from difflut.encoder import MyCustomEncoder

class TestMyCustomNode:
    def test_forward_shape(self):
        node = MyCustomNode(input_dim=[4], output_dim=[1])
        x = torch.randint(0, 2, (32, 4))
        output = node(x)
        assert output.shape == (32, 1)
    
    def test_gradients(self):
        node = MyCustomNode(input_dim=[4], output_dim=[1])
        x = torch.randint(0, 2, (32, 4), dtype=torch.float32)
        
        output = node(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert node.weights.grad is not None
        assert node.weights.grad.shape == node.weights.shape
    
    def test_regularization(self):
        node = MyCustomNode(input_dim=[4], output_dim=[1])
        reg = node.regularization()
        assert reg.item() > 0

class TestMyCustomEncoder:
    def test_fit_and_encode(self):
        encoder = MyCustomEncoder(num_bits=8)
        
        # Create training data
        train_data = torch.randn(100, 10)
        encoder.fit(train_data)
        
        # Encode
        test_data = torch.randn(32, 10)
        encoded = encoder(test_data)
        
        assert encoded.shape == (32, 10 * 8)
    
    def test_invalid_without_fit(self):
        encoder = MyCustomEncoder(num_bits=8)
        test_data = torch.randn(32, 10)
        
        with pytest.raises(RuntimeError):
            encoder(test_data)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_my_component.py::TestMyCustomNode::test_forward_shape

# Run with coverage
pytest --cov=difflut tests/
```

## Registering Your Component

Once implemented, components are automatically registered:

```python
from difflut.registry import get_node_class

# Get your registered component
NodeClass = get_node_class('my_custom_node')
node = NodeClass(input_dim=[4], output_dim=[1])
```

## Next Steps

1. **Review existing implementations** in `difflut/` for patterns
2. **Start simple** - begin with a basic component
3. **Add tests** - comprehensive test coverage
4. **Consider CUDA** - for performance-critical nodes
5. **Submit PR** - follow [Contributing Guide](contributing.md)

## Resources

- **Base Classes**: `difflut/nodes/base_node.py`, `difflut/encoder/base_encoder.py`, `difflut/layers/base_layer.py`
- **Examples**: Existing implementations in respective directories
- **Tests**: `tests/` directory
- **Registry**: `difflut/registry.py`
