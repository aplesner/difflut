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

```
Raw Input:      (batch_size, features)
    ↓
Encoder:        (batch_size, features * num_bits)
    ↓
Layer:          (batch_size, num_nodes)
    ↓
Layer:          (batch_size, num_classes)
```

**Example with MNIST (784 features → 10 classes)**:
```
Input:          (32, 784)               # 32 images, 784 pixels
    ↓
Encoder (8 bits): (32, 6272)            # 784 * 8 = 6272 encoded values
    ↓
Hidden Layer:   (32, 128)               # 128 hidden nodes
    ↓
Output Layer:   (32, 10)                # 10 class predictions
```

### Step 1: Import Dependencies

```python
import torch
import torch.nn as nn
from difflut.encoder import ThermometerEncoder
from difflut.layers import RandomLayer
from difflut.nodes import LinearLUTNode
```

### Step 2: Create a Simple LUT Network

```python
class SimpleLUTNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        
        # Step 2a: Encoder transforms continuous inputs to discrete values for LUT indexing
        # Input: (batch_size, 784) → Output: (batch_size, 784*8=6272)
        self.encoder = ThermometerEncoder(num_bits=8)
        encoded_size = input_size * 8  # 6272
        
        # Step 2b: Hidden layer with random connectivity
        # Input: (batch_size, 6272) → Output: (batch_size, 128)
        self.hidden = RandomLayer(
            input_size=encoded_size,        # 6272 input features
            output_size=hidden_size,        # 128 output nodes
            node_type=LinearLUTNode,
            n=4,  # Each LUT has 4 inputs
            node_kwargs={'input_dim': [4], 'output_dim': [1]}
        )
        
        # Step 2c: Output layer
        # Input: (batch_size, 128) → Output: (batch_size, 10)
        self.output = RandomLayer(
            input_size=hidden_size,         # 128 input features
            output_size=num_classes,        # 10 output nodes
            node_type=LinearLUTNode,
            n=4,  # Each LUT has 4 inputs
            node_kwargs={'input_dim': [4], 'output_dim': [1]}
        )
    
    def forward(self, x):
        # Input shape: (batch_size, 1, 28, 28) for MNIST
        
        # Flatten images and encode
        x = x.view(x.size(0), -1)           # → (batch_size, 784)
        x = self.encoder(x)                 # → (batch_size, 6272)
        
        # Pass through LUT layers with ReLU activation
        x = torch.relu(self.hidden(x))      # → (batch_size, 128)
        x = self.output(x)                  # → (batch_size, 10)
        
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
x_encoded = model.encoder(x_flat)       # torch.Size([32, 6272])
x_hidden = torch.relu(model.hidden(x_encoded))  # torch.Size([32, 128])
output = model.output(x_hidden)         # torch.Size([32, 10])

print(f"After encoder:      {x_encoded.shape}")  # torch.Size([32, 6272])
print(f"After hidden:       {x_hidden.shape}")   # torch.Size([32, 128])
print(f"Output shape:       {output.shape}")     # torch.Size([32, 10])

# Or use forward pass directly
output = model(x)

# Compute loss and backpropagate like normal
criterion = nn.CrossEntropyLoss()
loss = criterion(output, torch.randint(0, 10, (32,)))
loss.backward()
```

### Key Dimension Rules

✓ **Encoder output** = input features × num_bits  
✓ **Layer output** = output_size (number of nodes)  
✓ **Next layer input** = previous layer output  
❌ **Mismatch** = will raise clear error with expected/got dimensions



## Next Steps

### Train on MNIST

For a complete training example with real data, check out:
- `examples/mnist_difflut_tutorial.ipynb` - Jupyter notebook walkthrough

### Explore Other Components

DiffLUT provides many options to customize your network:

**Different Node Types**:
- `LinearLUTNode` - Simple linear LUT (fastest)
- `PolyLUTNode` - Polynomial LUT
- `NeuralLUTNode` - MLP-based LUT
- `FourierNode` - Fourier transform-based (GPU-accelerated)
- `HybridNode` - Hybrid approach (GPU-accelerated)

**Different Encoders**:
- `ThermometerEncoder` - Thermometer code
- `GrayEncoder` - Gray code
- `BinaryEncoder` - Binary representation
- `GaussianEncoder` - Gaussian distributions

**Different Layer Types**:
- `RandomLayer` - Random input connectivity
- `LearnableLayer` - Learns which inputs to use
- `GroupedLayer` - Semantic input grouping
- `ResidualLayer` - Skip connections for deeper networks

See [User Guide](USER_GUIDE.md) for comprehensive examples of each.

### Advanced Topics

- **Custom Components**: Learn how to implement [custom nodes, encoders, and layers](DEVELOPER_GUIDE/creating_components.md)
- **Registry System**: Discover how to use the [component registry](USER_GUIDE/registry_pipeline.md)
- **GPU Acceleration**: Use CUDA-accelerated nodes for better performance
- **FPGA Export**: Export trained networks for hardware deployment

## Common Patterns

### Using Different Encoders

```python
from difflut.encoder import GrayEncoder, BinaryEncoder

# Create and fit encoder on training data
encoder = GrayEncoder(num_bits=8, feature_wise=True)
encoder.fit(train_data)  # Fit to learn data statistics

# Use in your model
encoded = encoder(input_data)
```

### Mixing Node Types in One Network

```python
from difflut.nodes import PolyLUTNode, NeuralLUTNode

model = nn.Sequential(
    RandomLayer(input_size=256, output_size=128, 
                node_type=LinearLUTNode, n=4,
                node_kwargs={'input_dim': [4], 'output_dim': [1]}),
    nn.ReLU(),
    RandomLayer(input_size=128, output_size=64, 
                node_type=PolyLUTNode, n=6,
                node_kwargs={'input_dim': [6], 'output_dim': [1], 'degree': 3}),
    nn.ReLU(),
    RandomLayer(input_size=64, output_size=10, 
                node_type=LinearLUTNode, n=4,
                node_kwargs={'input_dim': [4], 'output_dim': [1]})
)
```

### Using Residual Layers for Deeper Networks

```python
from difflut.layers import ResidualLayer

# Stack residual LUT layers
residual_layer = ResidualLayer(
    input_size=256,
    output_size=256,
    base_layer_type=RandomLayer,
    node_type=LinearLUTNode,
    n=4,
    node_kwargs={'input_dim': [4], 'output_dim': [1]},
    residual_weight=0.5
)
```

## Tips & Best Practices

1. **Always fit encoders** on representative training data before using
2. **Start simple** with `LinearLUTNode` and `RandomLayer`
3. **Use appropriate `n` values** (typically 4-6 inputs per LUT)
4. **Batch size matters** for numerical stability in gradient computation
5. **GPU acceleration** available for Fourier, Hybrid, and other specialized nodes

## Troubleshooting

### Encoder not fitted error
```python
# ❌ Wrong: encoder not fitted
encoder = ThermometerEncoder(num_bits=8)
x = encoder(data)  # Error!

# ✓ Correct: fit first
encoder = ThermometerEncoder(num_bits=8)
encoder.fit(training_data)  # Fit to data
x = encoder(data)  # Now works
```

### Device mismatch errors
```python
# Ensure model and data are on same device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = x.to(device)
```

## More Information

- **Complete Usage Examples**: See [User Guide](USER_GUIDE.md)
- **Component Details**: Check [Components Guide](USER_GUIDE/components.md)
- **Registry & Pipelines**: Learn [registry system](USER_GUIDE/registry_pipeline.md)
- **For Developers**: Read [Developer Guide](DEVELOPER_GUIDE.md)
