"""
DiffLUT Models Directory

Structure:
    __init__.py        - Model registry and factory functions
    base_model.py      - Abstract base class for all models
    mnist.py           - MNIST classification models
    cifar10.py         - CIFAR-10 classification models (convolutional)
    comparison.py      - Models showcasing specific training techniques
    pretrained/        - (Future) Pre-trained model weights

## Quick API Reference

### Get Models

    from difflut import get_model, list_models
    
    # List all available models
    models = list_models()
    
    # Get a model
    model = get_model('mnist_fc_8k_linear')
    
    # Create directly
    from difflut.models import MNISTLinearSmall
    model = MNISTLinearSmall()

### Basic Workflow

    import torch
    from difflut import get_model
    
    # Create model
    model = get_model('mnist_fc_8k_linear')
    
    # Fit encoder on training data
    model.fit_encoder(train_images)  # Shape: (N, 28, 28) for MNIST
    
    # Forward pass
    outputs = model(test_images)  # Shape: (N, 10) logits
    
    # Training loop
    loss = criterion(model(batch), targets)
    reg_loss = model.get_regularization_loss()
    total_loss = loss + 0.001 * reg_loss
    total_loss.backward()

### Model Utilities

    # Count parameters
    counts = model.count_parameters()
    # {'total': 123456, 'trainable': 123456, 'non_trainable': 0}
    
    # Get layer topology
    topology = model.get_layer_topology()
    
    # Save/Load checkpoints
    model.save_checkpoint('model.pt')
    model.load_checkpoint('model.pt')

## Available Models

**MNIST Classification:**
- `mnist_fc_8k_linear`: Linear LUT nodes
- `mnist_fc_8k_dwn`: DWN stable nodes (GPU/FPGA friendly)

**CIFAR-10 Classification:**
- `cifar10_conv`: Convolutional + fully connected

**Training Technique Variants:**
- `mnist_bitflip_[0|5|10|20]`: Bit flipping robustness
- `mnist_gradnorm_[none|layerwise|batchwise|layerwise_mean]`: Gradient stabilization
- `mnist_residual_init`: Residual layer demonstration

## Design Philosophy

1. **Self-contained**: All parameters hardcoded for reproducibility
2. **Educational**: Demonstrate DiffLUT capabilities
3. **Production-ready**: Suitable for benchmarking and FPGA export
4. **Extensible**: Easy to add new models following BaseModel pattern

## Adding Custom Models

1. Create class inheriting from `BaseModel`
2. Implement required methods: `fit_encoder()`, `_build_layers()`, `_forward_impl()`
3. Add to registry in `__init__.py`
4. Write tests
5. Document in MODEL_ZOO.md

Example:
```python
from .base_model import BaseModel

class CustomModel(BaseModel):
    def __init__(self):
        super().__init__(name='custom', input_size=784, num_classes=10)
        # Initialize encoder and configuration
    
    def fit_encoder(self, data: torch.Tensor) -> None:
        # Fit encoder on data
        pass
    
    def _build_layers(self) -> None:
        # Build network layers
        pass
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass implementation
        pass
```

## Testing

Run model zoo tests:

    pytest tests/test_models/test_model_zoo.py -v

## Documentation

See `docs/MODEL_ZOO.md` for detailed documentation, examples, and workflow guides.
"""
