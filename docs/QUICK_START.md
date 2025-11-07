# Quick Start Guide

Get a DiffLUT LUT network running in 5 minutes!

## Installation

```bash
pip install -e .
```

For detailed setup instructions, see [Installation Guide](INSTALLATION.md).

## Your First LUT Network

### Understanding Dimensions

Before building, let's understand how data flows through DiffLUT:

**Example with MNIST (784 features → 10 classes)**:
```
Input:          (32, 784)               # 32 images, 784 pixels
    ↓
Encoder (8 bits): (32, 6272)            # 784 * 8 = 6272 encoded values
    ↓
First Layer:   (32, 1000)               # 1000 nodes
    ↓
Second Layer:   (32, 1000)               # 1000 nodes
    ↓
Group Sum:   (32, 10)                # 10 class predictions
```

### Step 1: Import Dependencies

```python
import torch
import torch.nn as nn
from difflut.encoder import ThermometerEncoder
from difflut.layers import RandomLayer
from difflut.layers.layer_config import LayerConfig
from difflut.nodes import LinearLUTNode
from difflut.nodes.node_config import NodeConfig
from difflut.utils.modules import GroupSum
```

### Step 2: Create a Simple LUT Network

```python
class SimpleLUTNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        
        # Step 2a: Encoder transforms continuous inputs to discrete values for LUT indexing
        # Input: (batch_size, 784) → Output: (batch_size, 784*8) automatically flattened
        # Note: flatten=True is the default, encoder automatically flattens to 2D
        self.encoder = ThermometerEncoder(num_bits=8)
        encoded_size = input_size * 8  # 6272 - encoder flattens automatically
        
        # Step 2b: Hidden layer with random connectivity
        # Create type-safe node configuration
        node_config = NodeConfig(
            input_dim=4,        # 4-input LUTs (each node processes (batch, 4) → (batch, 1))
            output_dim=1        # Single output per LUT
        )
        
        # Create type-safe layer configuration (optional, using defaults here)
        layer_config = LayerConfig(
            flip_probability=0.0,           # No bit flipping during training
            grad_stabilization='none'       # No gradient stabilization
        )
        
        # Input: (batch_size, 6272) → Output: (batch_size, 128)
        # Creates 128 independent LinearLUTNode instances in nn.ModuleList
        self.hidden = RandomLayer(
            input_size=encoded_size,        # 6272 input features
            output_size=hidden_size,        # 128 output nodes (128 independent LUTs)
            node_type=LinearLUTNode,
            n=4,                            # Each LUT has 4 inputs
            node_kwargs=node_config,
            layer_config=layer_config       # Training parameters
        )
        
        # Step 2c: Output layer routes to num_classes nodes
        # Input: (batch_size, 128) → Output: (batch_size, 10)
        # Creates 10 independent LinearLUTNode instances in nn.ModuleList
        self.output = RandomLayer(
            input_size=hidden_size,         # 128 input features
            output_size=num_classes,        # 10 output nodes (10 independent LUTs)
            node_type=LinearLUTNode,
            n=4,                            # Each LUT has 4 inputs
            node_kwargs=node_config,
            layer_config=layer_config       # Training parameters
        )
        
        # Step 2d: GroupSum groups output features and sums them
        # Input: (batch_size, 10) → Output: (batch_size, num_classes=10)
        # Groups the 10 outputs into num_classes groups and sums within each group
        self.groupsum = GroupSum(k=num_classes, tau=1.0)
    
    def forward(self, x):
        # Input shape: (batch_size, 1, 28, 28) for MNIST
        
        # Flatten images and encode
        x = x.view(x.size(0), -1)           # → (batch_size, 784)
        x = self.encoder(x)                 # → (batch_size, 6272) - auto-flattened
        
        # Pass through LUT layers with ReLU activation
        x = torch.relu(self.hidden(x))      # → (batch_size, 128)
        x = self.output(x)                  # → (batch_size, 10)
        
        # Group and sum outputs to get final predictions
        x = self.groupsum(x)                # → (batch_size, 10)
        
        return x
```

### Step 3: Use Your Network

```python
# Create model and dummy input (MNIST-like)
model = SimpleLUTNetwork()
x = torch.randn(32, 1, 28, 28)  # Batch of 32 MNIST images

# Forward pass - watch the dimensions
print(f"Input shape:        {x.shape}")  # torch.Size([32, 1, 28, 28])

# Manually trace through to see dimensions
x_flat = x.view(x.size(0), -1)          # torch.Size([32, 784])
x_encoded = model.encoder(x_flat)       # torch.Size([32, 6272]) - auto-flattened
x_hidden = torch.relu(model.hidden(x_encoded))  # torch.Size([32, 128])
x_output = model.output(x_hidden)       # torch.Size([32, 10])
predictions = model.groupsum(x_output)  # torch.Size([32, 10])

print(f"After encoder:      {x_encoded.shape}")      # torch.Size([32, 6272])
print(f"After hidden:       {x_hidden.shape}")       # torch.Size([32, 128])
print(f"After output layer: {x_output.shape}")       # torch.Size([32, 10])
print(f"After groupsum:     {predictions.shape}")    # torch.Size([32, 10])

# Or use forward pass directly
predictions = model(x)

# Compute loss and backpropagate like normal
criterion = nn.CrossEntropyLoss()
loss = criterion(predictions, torch.randint(0, 10, (32,)))
loss.backward()
```

### Key Dimension Rules

✓ **Encoder input** = (batch_size, input_features)  
✓ **Encoder output (auto-flattened)** = (batch_size, input_features × num_bits)  
✓ **Layer output** = (batch_size, output_size) - Each layer has output_size independent nodes  
✓ **Node processing** = (batch_size, input_dim) → (batch_size, output_dim) - 2D tensors only  
✓ **GroupSum input** = (batch_size, num_nodes)  
✓ **GroupSum output** = (batch_size, k_groups)  
❌ **Mismatch** = will raise clear error with expected/got dimensions

**Architecture Note**: Layers use `nn.ModuleList` containing `output_size` independent node instances. Each node processes 2D tensors. The layer processes nodes efficiently using preallocated output tensors.



## Next Steps

### Train on MNIST

For a complete training example with real data, check out:
- `examples/mnist_difflut_tutorial.ipynb` - Jupyter notebook walkthrough

### Explore Other Components

DiffLUT provides many options to customize your network. See [User Guide](USER_GUIDE.md) for comprehensive examples of each component.

### Advanced Topics

- **Custom Components**: Learn how to implement [custom nodes, encoders, and layers](DEVELOPER_GUIDE/creating_components.md)
- **Registry System**: Discover how to use the [component registry](USER_GUIDE/registry_pipeline.md)
- **GPU Acceleration**: Use CUDA-accelerated nodes for better performance
- **FPGA Export**: Export trained networks for hardware deployment

