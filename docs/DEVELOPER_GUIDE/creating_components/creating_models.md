# Creating Custom Models

Learn how to implement custom DiffLUT model architectures, organize pretrained models, and integrate with the model registry system.

---

## Quick Start

### Create a Simple Model

```python
import torch.nn as nn
from difflut.models import BaseLUTModel, ModelConfig
from difflut import register_model

@register_model('my_feedforward')
class MyFeedForward(BaseLUTModel):
    """Custom feedforward model."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._build_layers()
    
    def _build_layers(self):
        """Build model layers."""
        # Build from config
        self.layers = nn.ModuleList()
        # ... layer construction ...
    
    def fit_encoder(self, data: torch.Tensor) -> None:
        """Fit encoder on training data."""
        self.encoder.fit(data)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementation."""
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        return x
```

Then use it:

```python
from difflut.models import build_model

config = ModelConfig(model_type='my_feedforward', ...)
model = build_model(config)
```

---

## Base Classes

### BaseLUTModel

All DiffLUT models inherit from `BaseLUTModel`:

```python
from difflut.models import BaseLUTModel, ModelConfig

class MyModel(BaseLUTModel):
    """Base class for all DiffLUT models."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize model.
        
        Args:
            config: ModelConfig instance with model parameters
        """
        super().__init__(config)
    
    def fit_encoder(self, data: torch.Tensor) -> None:
        """
        Fit encoder on training data.
        
        Args:
            data: Training data of shape (N, features)
        """
        raise NotImplementedError
    
    def _build_layers(self) -> None:
        """Build model layers from config."""
        raise NotImplementedError
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        Actual forward pass (after encoder).
        
        Args:
            x: Encoded input
        
        Returns:
            Model output
        """
        raise NotImplementedError
```

### ModelConfig

Dataclass for organizing model configuration:

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class ModelConfig:
    # Structural parameters
    model_type: str              # Model architecture type
    layer_type: str              # Layer implementation
    node_type: str               # Node implementation
    encoder_config: Dict[str, Any]  # Encoder parameters
    node_input_dim: int          # Bits per node
    layer_widths: List[int]      # Hidden layer sizes
    num_classes: int             # Output classes
    
    # Optional parameters
    input_size: Optional[int] = None
    dataset: Optional[str] = None
    seed: int = 42
    
    # Runtime parameters (safe to override)
    runtime: Dict[str, Any] = field(default_factory=dict)
    
    # Pretrained info
    pretrained: bool = False
    pretrained_name: Optional[str] = None
    
    def save_to_pretrained(
        self,
        name: str,
        pretrained_dir: str = "pretrained"
    ) -> Path:
        """Save config to pretrained directory."""
        # Implementation creates pretrained/<model_type>/name.yaml
        pass
```

---

## Creating a Complete Model

### Example: Feedforward Model

```python
import torch
import torch.nn as nn
from typing import List, Optional
from difflut.models import BaseLUTModel, ModelConfig
from difflut.registry import REGISTRY
from difflut import register_model

@register_model('simple_feedforward')
class SimpleFeedForward(BaseLUTModel):
    """Simple feedforward LUT-based model."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize feedforward model.
        
        Args:
            config: ModelConfig instance
        """
        super().__init__(config)
        self.config = config
        
        # Get encoder from registry
        encoder_name = config.encoder_config.get('name', 'thermometer')
        EncoderClass = REGISTRY.get_encoder(encoder_name)
        self.encoder = EncoderClass(**config.encoder_config)
        
        # Build layers
        self._build_layers()
    
    def _build_layers(self) -> None:
        """Build fully connected LUT layers."""
        self.layers = nn.ModuleList()
        
        # Calculate input size after encoding
        encoder_name = self.config.encoder_config.get('name', 'thermometer')
        if encoder_name == 'thermometer':
            num_bits = self.config.encoder_config.get('num_bits', 4)
            input_size = (self.config.input_size or 784) * num_bits
        else:
            input_size = self.config.input_size or 784
        
        # Build layers
        prev_size = input_size
        for width in self.config.layer_widths:
            LayerClass = REGISTRY.get_layer(self.config.layer_type)
            layer = LayerClass(
                input_size=prev_size,
                output_size=width,
                node_type=self.config.node_type,
                n=self.config.node_input_dim,
                node_kwargs=self._get_node_kwargs(),
            )
            self.layers.append(layer)
            prev_size = width
        
        # Output layer
        OutputLayerClass = REGISTRY.get_layer(self.config.layer_type)
        output_layer = OutputLayerClass(
            input_size=prev_size,
            output_size=self.config.num_classes,
            node_type=self.config.node_type,
            n=self.config.node_input_dim,
            node_kwargs=self._get_node_kwargs(),
        )
        self.layers.append(output_layer)
    
    def _get_node_kwargs(self) -> Dict[str, Any]:
        """Get node configuration from runtime parameters."""
        runtime = self.config.runtime or {}
        return {
            'temperature': runtime.get('temperature', 1.0),
            'eval_mode': runtime.get('eval_mode', 'expectation'),
        }
    
    def fit_encoder(self, data: torch.Tensor) -> None:
        """
        Fit encoder on training data.
        
        Args:
            data: Training data of shape (N, features)
        """
        if len(data.shape) > 2:
            # Flatten images
            data = data.reshape(data.shape[0], -1)
        
        self.encoder.fit(data)
        self._is_encoder_fitted = True
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through layers.
        
        Args:
            x: Encoded input
        
        Returns:
            Model output logits
        """
        for layer in self.layers:
            x = layer(x)
        return x
```

### Complete Training Example

```python
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from difflut.models import build_model, ModelConfig

# Create model config
config = ModelConfig(
    model_type='simple_feedforward',
    layer_type='random',
    node_type='probabilistic',
    encoder_config={'name': 'thermometer', 'num_bits': 4},
    node_input_dim=6,
    layer_widths=[1024, 1000],
    num_classes=10,
    input_size=784,
    dataset='mnist'
)

# Build model
model = build_model(config)

# Fit encoder
train_loader = DataLoader(train_dataset, batch_size=128)
all_train_data = torch.cat([batch for batch, _ in train_loader])
model.fit_encoder(all_train_data)

# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

for epoch in range(50):
    model.train()
    for batch, targets in train_loader:
        batch = batch.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, targets)
        
        # Add regularization
        reg_loss = model.get_regularization_loss()
        total_loss = loss + 0.001 * reg_loss
        
        total_loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}: done")
```

---

## Model Configuration

### Structural vs. Runtime Parameters

**Structural Parameters** (define architecture, cannot change with pretrained weights):
- `model_type`: Model class
- `layer_type`: Layer implementation
- `node_type`: Node implementation
- `encoder_config`: Encoder parameters
- `node_input_dim`: LUT input dimension
- `layer_widths`: Hidden layer sizes
- `num_classes`: Output classes

**Runtime Parameters** (can be overridden safely):
- `temperature`: Probabilistic node temperature
- `eval_mode`: Expectation or sampling
- `flip_probability`: Bit flip probability
- `regularizers`: Regularization weights
- Any other non-structural parameter in `runtime` dict

### Configuration File Format (YAML)

```yaml
# Structural parameters
model_type: simple_feedforward
layer_type: random
node_type: probabilistic

encoder_config:
  name: thermometer
  num_bits: 4
  feature_wise: true

node_input_dim: 6
layer_widths: [1024, 1000]
num_classes: 10
input_size: 784
dataset: mnist

# Optional
seed: 42

# Runtime parameters (can be overridden)
runtime:
  temperature: 1.0
  eval_mode: expectation
  flip_probability: 0.0
  regularizers:
    entropy: 0.001

# Pretrained info
pretrained: false
pretrained_name: null
```

### Loading from YAML

```python
from difflut.models import build_model, ModelConfig
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

config = ModelConfig(**config_dict)

# Build model
model = build_model(config)
```

---

## Registering and Discovering Models

### Registering a Model

Use the `@register_model` decorator:

```python
from difflut import register_model
from difflut.models import BaseLUTModel

@register_model('my_model_name')
class MyModel(BaseLUTModel):
    """Model description."""
    pass
```

### Discovering Registered Models

```python
from difflut.registry import REGISTRY

# List all registered models
print(REGISTRY.list_models())

# Get a model class
ModelClass = REGISTRY.get_model('my_model_name')

# Check if registered
if 'my_model_name' in REGISTRY.list_models():
    print("Model is registered")
```

### Using via Registry

```python
from difflut.models import build_model, ModelConfig
from difflut.registry import REGISTRY

# Via build_model (automatic registry lookup)
model = build_model('my_model_name')

# Or explicit registry lookup
ModelClass = REGISTRY.get_model('my_model_name')
config = ModelConfig(model_type='my_model_name', ...)
model = ModelClass(config)
```

---

## Pretrained Models

### Directory Structure

```
pretrained/
├── feedforward/
│   ├── mnist_large.yaml
│   ├── mnist_large.pth
│   ├── mnist_small.yaml
│   ├── mnist_small.pth
│   └── cifar10_large.yaml
├── convnet/
│   ├── cifar10_conv.yaml
│   └── cifar10_conv.pth
└── README.md
```

### Creating a Pretrained Model

**Step 1: Train your model**

```python
from difflut.models import build_model, ModelConfig

config = ModelConfig(
    model_type='feedforward',
    layer_type='random',
    node_type='probabilistic',
    encoder_config={'name': 'thermometer', 'num_bits': 4},
    node_input_dim=6,
    layer_widths=[1024, 1000],
    num_classes=10,
    dataset='mnist',
    pretrained=True,
    pretrained_name='mnist_large'
)

model = build_model(config)
model.fit_encoder(train_data)

# Train model...
# (training code here)
```

**Step 2: Save config and weights**

```python
from pathlib import Path

# Create directory
pretrained_dir = Path("difflut/models/pretrained/feedforward")
pretrained_dir.mkdir(parents=True, exist_ok=True)

# Save config
model.save_config(str(pretrained_dir / "mnist_large.yaml"))

# Save weights
model.save_weights(str(pretrained_dir / "mnist_large.pth"))
```

**Step 3: Update config with metadata**

Edit the saved YAML to include:

```yaml
# Add training metadata
accuracy: 0.975
training_time: 3600  # seconds
training_dataset: mnist_60k
training_date: "2025-01-15"

# Ensure pretrained info is set
pretrained: true
pretrained_name: mnist_large
```

**Step 4: Use the model**

```python
from difflut.models import build_model, list_pretrained_models

# Check it's available
print("mnist_large" in list_pretrained_models()["feedforward"])

# Load and use
model = build_model("mnist_large", load_weights=True)
predictions = model(test_data)
```

### Loading Pretrained Models

```python
from difflut.models import build_model, get_pretrained_model_info

# List available models
from difflut.models import list_pretrained_models
print(list_pretrained_models())

# Get info about a model
info = get_pretrained_model_info("mnist_large")
print(f"Accuracy: {info.get('accuracy')}")
print(f"Architecture: {info['model_type']}")

# Load with weights
model = build_model("mnist_large", load_weights=True)

# Load architecture only (for fine-tuning)
model = build_model("mnist_large", load_weights=False)
```

### Organizing Pretrained Models

Best practices for pretrained model organization:

1. **Naming**: `{dataset}_{architecture}_{size}`
   - Good: `mnist_large`, `cifar10_small`, `imagenet32_medium`
   - Bad: `model1`, `final`, `best_v2`

2. **Directory**: One directory per model type
   - `pretrained/feedforward/`: All feedforward models
   - `pretrained/convnet/`: All convolutional models
   - `pretrained/residual/`: All residual models

3. **Metadata in YAML**: Include training details

```yaml
# Training metadata
accuracy: 0.975
test_accuracy: 0.972
training_loss_final: 0.0845

# Hardware info
training_device: "GPU A100"
training_time_hours: 1.5

# Model info
parameters_total: 1_234_567
parameters_trainable: 1_234_500

# Dates
training_date: "2025-01-15"
validation_date: "2025-01-16"

# Notes
notes: "Trained with bit flip probability 0.05"
```

4. **README in directory**:

```markdown
# Pretrained Feedforward Models

## Available Models

- `mnist_large`: 98.5% accuracy on MNIST
- `mnist_small`: 97.2% accuracy on MNIST (faster)
- `cifar10_large`: 75.3% accuracy on CIFAR-10

## Usage

See [MODEL_USAGE_GUIDE.md](../MODEL_USAGE_GUIDE.md)
```

---

## Advanced Patterns

### Conditional Layer Construction

```python
class FlexibleModel(BaseLUTModel):
    def _build_layers(self) -> None:
        """Build layers based on config."""
        self.layers = nn.ModuleList()
        
        # Use different layer types based on config
        for i, width in enumerate(self.config.layer_widths):
            if i == 0 and self.config.runtime.get('use_residual'):
                # Use residual layer
                LayerClass = REGISTRY.get_layer('residual')
            else:
                # Use standard layer
                LayerClass = REGISTRY.get_layer(self.config.layer_type)
            
            layer = LayerClass(...)
            self.layers.append(layer)
```

### Model Composition

```python
class CompositeModel(BaseLUTModel):
    """Model combining multiple submodels."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Create multiple branches
        self.branch1 = self._build_branch(config, head_size=128)
        self.branch2 = self._build_branch(config, head_size=64)
        
        # Merge layer
        self.merge = nn.Linear(128 + 64, config.num_classes)
    
    def _build_branch(self, config, head_size):
        """Build a single branch."""
        layers = nn.ModuleList()
        # ... build layers ...
        return layers
    
    def _forward_impl(self, x):
        x1 = x
        for layer in self.branch1:
            x1 = layer(x1)
        
        x2 = x
        for layer in self.branch2:
            x2 = layer(x2)
        
        merged = torch.cat([x1, x2], dim=1)
        return self.merge(merged)
```

### Dynamic Configuration

```python
class AdaptiveModel(BaseLUTModel):
    """Model that adapts to input size."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._encoder_fitted = False
        self.input_size = None
    
    def fit_encoder(self, data: torch.Tensor) -> None:
        """Fit encoder and build layers based on encoded size."""
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        
        self.encoder.fit(data)
        self._encoder_fitted = True
        
        # Get encoded size
        encoded_sample = self.encoder(data[:1])
        self.input_size = encoded_sample.shape[1]
        
        # Now build layers with correct input size
        self._build_layers()
    
    def _build_layers(self) -> None:
        """Build layers using discovered input size."""
        if not self._encoder_fitted or self.input_size is None:
            raise RuntimeError("Must fit encoder first")
        
        # Build using self.input_size
        # ...
```

---

## Testing Custom Models

```python
import torch
import pytest
from difflut.models import build_model, ModelConfig

class TestMyModel:
    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        config = ModelConfig(
            model_type='my_model',
            layer_type='random',
            node_type='probabilistic',
            encoder_config={'name': 'thermometer', 'num_bits': 4},
            node_input_dim=6,
            layer_widths=[128, 64],
            num_classes=10,
            input_size=784
        )
        
        model = build_model(config)
        model.fit_encoder(torch.randn(100, 784))
        
        output = model(torch.randn(32, 784))
        assert output.shape == (32, 10)
    
    def test_encoder_required(self):
        """Test that encoder must be fitted."""
        config = ModelConfig(...)
        model = build_model(config)
        
        with pytest.raises(RuntimeError):
            model(torch.randn(32, 784))
    
    def test_device_handling(self):
        """Test GPU/CPU device handling."""
        config = ModelConfig(...)
        model = build_model(config)
        model.fit_encoder(torch.randn(100, 784))
        
        # CPU
        output_cpu = model(torch.randn(32, 784))
        
        # GPU (if available)
        if torch.cuda.is_available():
            model = model.cuda()
            output_gpu = model(torch.randn(32, 784).cuda())
            assert output_gpu.device.type == 'cuda'
```

---

## Integration with Registry and Pipelines

### Using with Build Pipelines

```python
from difflut.models import build_model
from difflut.registry import REGISTRY

# Via registry
ModelClass = REGISTRY.get_model('my_model')
model = ModelClass(config)

# Via factory
model = build_model(config)

# Via name string
model = build_model('my_model_name')
```

### Integrating with Experiment Configs

Models integrate with Hydra experiment configs:

```yaml
# experiments/config/mnist.yaml
model:
  model_type: simple_feedforward
  layer_type: random
  node_type: probabilistic
  encoder_config:
    name: thermometer
    num_bits: 4
  node_input_dim: 6
  layer_widths: [1024, 1000]
  num_classes: 10

training:
  epochs: 50
  learning_rate: 0.001
```

Then in your experiment script:

```python
from omegaconf import DictConfig
from difflut.models import build_model, ModelConfig

def run_experiment(cfg: DictConfig):
    # Build model from config
    model_config = ModelConfig(**cfg.model)
    model = build_model(model_config)
    
    # Train...
```

---

## Best Practices

1. **Inherit from BaseLUTModel**: Ensures compatibility with model system
2. **Use Registry**: Register layers, nodes, encoders via decorators
3. **Type Hints**: Full PEP 484 type hints for clarity
4. **Documentation**: NumPy-style docstrings on all methods
5. **Configuration**: Separate structural and runtime parameters
6. **Testing**: Comprehensive unit tests for all models
7. **Pretrained**: Organize in `pretrained/<model_type>/` directories
8. **Metadata**: Include training details in config YAML
9. **Device Handling**: Support CPU/GPU transparently
10. **Error Messages**: Clear RuntimeError messages for user mistakes

---

## Next Steps

- **Use Models**: See [Using DiffLUT Models](../../USER_GUIDE/components/models.md)
- **Build Pipelines**: See [Registry & Pipelines](../../USER_GUIDE/registry_pipeline.md)
- **Export**: See [Export Guide](../../USER_GUIDE/export_guide.md)
- **Examples**: See `examples/` directory for complete examples
