# DiffLUT Models Module

A unified, config-based system for creating, managing, and deploying DiffLUT models with pretrained weight support and runtime parameter overrides.

## 🎯 Key Features

- **Config-Based Architecture**: Clear separation of structural and runtime parameters
- **Pretrained Model Support**: Easy loading and sharing of trained models
- **Runtime Overrides**: Safe parameter changes without retraining
- **Flexible Building**: Multiple ways to create models (name, YAML, config object)
- **Registry Integration**: Automatic discovery and registration
- **Type Safety**: Dataclass-based configuration with validation
- **YAML Serialization**: Human-readable config files

## 📁 Module Structure

```
models/
├── __init__.py              # Main exports
├── base_model.py            # BaseLUTModel base class
├── model_config.py          # ModelConfig dataclass
├── feedforward.py           # SimpleFeedForward implementation
├── factory.py               # build_model and loading functions
├── MODEL_USAGE_GUIDE.md     # Detailed usage guide
└── pretrained/              # Pretrained model directory
    ├── README.md
    └── feedforward/
        └── mnist_large.yaml # Example pretrained config
```

## 🚀 Quick Start

### Load a Pretrained Model

```python
from difflut.models import build_model

model = build_model("mnist_large", load_weights=True)
output = model(input_data)
```

### Build from Config

```python
from difflut.models import ModelConfig, SimpleFeedForward

config = ModelConfig(
    model_type="feedforward",
    layer_type="random",
    node_type="probabilistic",
    encoder_config={"name": "thermometer", "num_bits": 4},
    node_input_dim=6,
    layer_widths=[1024, 1000],
    num_classes=10
)

model = SimpleFeedForward(config)
model.fit_encoder(train_data)
```

### Override Runtime Parameters

```python
model = build_model(
    "mnist_large",
    overrides={
        "temperature": 0.5,
        "eval_mode": "sampling"
    }
)
```

## 📖 Core Components

### ModelConfig

Dataclass for model configuration with structural and runtime parameters:

```python
@dataclass
class ModelConfig:
    # Structural (must match pretrained weights)
    model_type: str
    layer_type: str
    node_type: str
    encoder_config: Dict[str, Any]
    node_input_dim: int
    layer_widths: List[int]
    num_classes: int
    
    # Runtime (safe to override)
    runtime: Dict[str, Any] = field(default_factory=dict)
    
    # Pretrained info
    pretrained: bool = False
    pretrained_name: Optional[str] = None
```

### BaseLUTModel

Base class for all DiffLUT models:

```python
class BaseLUTModel(nn.Module):
    def __init__(self, config: ModelConfig)
    def forward(self, x: torch.Tensor) -> torch.Tensor
    def apply_runtime_overrides(self, overrides: Dict[str, Any])
    def get_regularization_loss(self) -> torch.Tensor
    def save_config(self, path: str)
    def save_weights(self, path: str)
    def load_weights(self, path: str)
```

### SimpleFeedForward

Clean feedforward network implementation:

```python
class SimpleFeedForward(BaseLUTModel):
    def __init__(self, config: ModelConfig)
    def fit_encoder(self, data: torch.Tensor)
    def encode(self, x: torch.Tensor) -> torch.Tensor
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

### Factory Functions

```python
# Main factory function
build_model(source, load_weights=True, overrides=None)

# Convenience functions
load_pretrained(name, **kwargs)
list_pretrained_models()
get_pretrained_model_info(name)
```

## 🏗️ Usage Patterns

### Pattern 1: Research with Pretrained Models

```python
# Load baseline model
baseline = build_model("mnist_large", load_weights=True)

# Test with different temperatures
for temp in [0.1, 0.5, 1.0, 2.0]:
    model = build_model("mnist_large", overrides={"temperature": temp})
    accuracy = evaluate(model, test_loader)
    print(f"Temp {temp}: {accuracy:.2%}")
```

### Pattern 2: Training from Scratch

```python
# Create config
config = ModelConfig.from_yaml("configs/my_experiment.yaml")

# Build and train
model = build_model(config)
model.fit_encoder(train_data)
train_model(model, train_loader, num_epochs=50)

# Save results
model.save_config("results/my_model.yaml")
model.save_weights("results/my_model.pth")
```

### Pattern 3: Hyperparameter Sweeps

```python
import itertools

# Define search space
layer_widths = [[512, 512], [1024, 1000], [2048, 2048]]
temperatures = [0.5, 1.0, 2.0]

# Run sweep
for widths, temp in itertools.product(layer_widths, temperatures):
    config = ModelConfig(
        model_type="feedforward",
        layer_widths=widths,
        runtime={"temperature": temp},
        ...
    )
    model = build_model(config)
    # Train and evaluate...
```

### Pattern 4: Transfer Learning

```python
# Load pretrained model
source_model = build_model("mnist_large", load_weights=True)

# Create new model with different output size
target_config = source_model.get_config()
target_config.num_classes = 100  # New task
target_config.pretrained = False

target_model = build_model(target_config, load_weights=False)

# Copy weights for shared layers
# (implementation depends on model architecture)
```

## 🔧 Configuration Guidelines

### Structural Parameters

These define the model architecture and **cannot be changed** when loading pretrained weights:

- `model_type`: Model architecture type
- `layer_type`: Layer implementation
- `node_type`: Node implementation
- `encoder_config`: Encoder configuration
- `node_input_dim`: Node input dimension
- `layer_widths`: Hidden layer sizes
- `num_classes`: Output classes
- `input_size`: Input feature size

### Runtime Parameters

These can be safely overridden without affecting weight compatibility:

- `temperature`: Temperature for probabilistic nodes
- `eval_mode`: Evaluation mode (expectation, sampling)
- `flip_probability`: Bit flip probability for testing
- `dropout`: Dropout rate (if applicable)
- `grad_stabilization`: Gradient stabilization method
- Regularizer weights
- Initialization parameters (for training)

### Example Config (YAML)

```yaml
# Structural parameters
model_type: feedforward
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

# Runtime parameters
runtime:
  temperature: 1.0
  eval_mode: expectation
  init_fn: xavier_uniform
  regularizers:
    entropy: {weight: 0.001}
    clarity: {weight: 0.0001}

# Pretrained info
pretrained: true
pretrained_name: mnist_large
```

## 📦 Pretrained Models

### Directory Structure

```
pretrained/
├── feedforward/
│   ├── mnist_large.yaml
│   ├── mnist_large.pth
│   ├── mnist_small.yaml
│   └── cifar10_large.yaml
└── convnet/
    └── ...
```

### Creating a Pretrained Model

1. Train your model:

```python
model = SimpleFeedForward(config)
model.fit_encoder(train_data)
train_model(model)
```

2. Save config and weights:

```python
config = model.get_config()
config.pretrained = True
config.pretrained_name = "my_model"

model.save_config("pretrained/feedforward/my_model.yaml")
model.save_weights("pretrained/feedforward/my_model.pth")
```

3. Use it:

```python
model = build_model("my_model", load_weights=True)
```

## 🧪 Testing

Run the example script:

```bash
cd difflut
python examples/model_system_example.py
```

## 📚 Documentation

- **MODEL_USAGE_GUIDE.md**: Comprehensive usage guide with examples
- **pretrained/README.md**: Guide for pretrained models
- **API Documentation**: See docstrings in source files

## 🔄 Migration from Old System

### Old Way (LayeredFeedForward)

```python
from experiments.models.layered_feedforward import LayeredFeedForward

config = {
    'encoder': {'name': 'thermometer', 'num_bits': 4},
    'layer_type': 'random',
    'node_type': 'probabilistic',
    'node_input_dim': 6,
    'hidden_sizes': [1024, 1000],
    'node_params': {...}
}
model = LayeredFeedForward(config, input_size=784, num_classes=10)
```

### New Way (SimpleFeedForward)

```python
from difflut.models import build_model

model = build_model("configs/my_model.yaml")
```

## 🤝 Contributing

When adding new models:

1. Inherit from `BaseLUTModel`
2. Use `ModelConfig` for configuration
3. Register with `@REGISTRY.register_model("name")`
4. Document in MODEL_USAGE_GUIDE.md
5. Add example config to pretrained/

## 📝 License

Same as parent DiffLUT project.

## ❓ Support

For questions or issues:
1. Check MODEL_USAGE_GUIDE.md
2. See examples/model_system_example.py
3. Check existing configs in pretrained/
4. File an issue on GitHub

## 🎓 Examples

See `examples/model_system_example.py` for:
- Creating models from config
- Loading pretrained models
- Using runtime overrides
- Saving and loading
- Building from YAML
- And more!

## 🔗 Related Components

- **Registry**: `difflut.registry.REGISTRY`
- **Nodes**: `difflut.nodes`
- **Layers**: `difflut.layers`
- **Encoders**: `difflut.encoders`
- **Utils**: `difflut.utils`
