# DiffLUT Model System - Usage Guide

This guide explains how to use the new DiffLUT model system with config-based architecture, pretrained models, and runtime parameter overrides.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Model Configuration](#model-configuration)
3. [Building Models](#building-models)
4. [Pretrained Models](#pretrained-models)
5. [Runtime Overrides](#runtime-overrides)
6. [Training a Model](#training-a-model)
7. [Saving and Loading](#saving-and-loading)

---

## Quick Start

### Load a Pretrained Model

```python
from difflut.models import build_model

# Load pretrained model with weights
model = build_model("mnist_large", load_weights=True)

# Use for inference
output = model(input_data)
```

### Build from Config File

```python
# Build from YAML config
model = build_model("configs/my_model.yaml")
```

### Create Custom Model

```python
from difflut.models import ModelConfig, SimpleFeedForward

config = ModelConfig(
    model_type="feedforward",
    layer_type="random",
    node_type="probabilistic",
    encoder_config={
        "name": "thermometer",
        "num_bits": 4,
        "feature_wise": True
    },
    node_input_dim=6,
    layer_widths=[1024, 1000],
    num_classes=10,
    dataset="mnist"
)

model = SimpleFeedForward(config)
```

---

## Model Configuration

### ModelConfig Structure

The `ModelConfig` dataclass separates **structural parameters** (which define architecture) from **runtime parameters** (which can be safely overridden).

```python
@dataclass
class ModelConfig:
    # Structural parameters (must match pretrained weights)
    model_type: str              # "feedforward", "convnet", etc.
    layer_type: str              # "random", "residual", etc.
    node_type: str               # "probabilistic", "dwn", etc.
    encoder_config: Dict         # Encoder configuration
    node_input_dim: int          # Input dimension for nodes
    layer_widths: List[int]      # Width of each layer
    num_classes: int             # Number of output classes
    
    # Optional structural parameters
    input_size: Optional[int]    # Raw input size (before encoding)
    dataset: Optional[str]       # Dataset name (for reference)
    seed: int = 42              # Random seed
    
    # Runtime parameters (safe to override)
    runtime: Dict[str, Any] = field(default_factory=dict)
    
    # Pretrained information
    pretrained: bool = False
    pretrained_name: Optional[str] = None
```

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
seed: 42

# Runtime parameters
runtime:
  temperature: 1.0
  eval_mode: expectation
  use_cuda: true
  
  init_fn: xavier_uniform
  init_v_target: 0.25
  
  regularizers:
    entropy:
      weight: 0.001
    clarity:
      weight: 0.0001
  
  flip_probability: 0.0
  grad_stabilization: none

# Pretrained info
pretrained: true
pretrained_name: mnist_large
```

---

## Building Models

### Method 1: Pretrained Model Name

```python
from difflut.models import build_model

# Load with weights
model = build_model("mnist_large", load_weights=True)

# Load architecture only (for training)
model = build_model("mnist_large", load_weights=False)
```

### Method 2: YAML Config File

```python
# Build from YAML
model = build_model("configs/my_model.yaml")
```

### Method 3: ModelConfig Object

```python
from difflut.models import ModelConfig, build_model

config = ModelConfig(
    model_type="feedforward",
    layer_type="random",
    node_type="probabilistic",
    encoder_config={"name": "thermometer", "num_bits": 4},
    node_input_dim=6,
    layer_widths=[1024, 1000],
    num_classes=10
)

model = build_model(config)
```

### Method 4: Direct Instantiation

```python
from difflut.models import SimpleFeedForward, ModelConfig

config = ModelConfig(...)
model = SimpleFeedForward(config)
```

---

## Pretrained Models

### List Available Pretrained Models

```python
from difflut.models import list_pretrained_models

models = list_pretrained_models()
# Returns: {"feedforward": ["mnist_large", "mnist_small"], ...}
```

### Get Model Information

```python
from difflut.models import get_pretrained_model_info

info = get_pretrained_model_info("mnist_large")
print(info['config'])
print(info['has_weights'])
print(info['model_type'])
print(info['layer_widths'])
```

### Load Pretrained Model (Convenience Function)

```python
from difflut.models import load_pretrained

# Equivalent to build_model(name, load_weights=True)
model = load_pretrained("mnist_large")
```

---

## Runtime Overrides

Runtime parameters can be safely overridden without affecting model structure or pretrained weight compatibility.

### Override at Build Time

```python
# Override temperature and eval mode
model = build_model(
    "mnist_large",
    overrides={
        "temperature": 0.5,
        "eval_mode": "sampling",
        "flip_probability": 0.01
    }
)
```

### Override After Creation

```python
# Create model
model = build_model("mnist_large")

# Apply runtime overrides
model.apply_runtime_overrides({
    "temperature": 0.5,
    "eval_mode": "sampling"
})
```

### Common Runtime Parameters

```python
runtime_params = {
    # Probabilistic node parameters
    "temperature": 1.0,
    "eval_mode": "expectation",  # or "sampling"
    "use_cuda": True,
    
    # Initialization (for training)
    "init_fn": "xavier_uniform",
    "init_v_target": 0.25,
    
    # Regularization (for training)
    "regularizers": {
        "entropy": {"weight": 0.001},
        "clarity": {"weight": 0.0001}
    },
    
    # Layer parameters
    "flip_probability": 0.0,
    "grad_stabilization": "none",
    "grad_target_std": 1.0,
    
    # Output layer
    "output_tau": 1.0
}
```

---

## Training a Model

### Complete Training Example

```python
import torch
from torch.utils.data import DataLoader
from difflut.models import build_model, ModelConfig

# 1. Create configuration
config = ModelConfig(
    model_type="feedforward",
    layer_type="random",
    node_type="probabilistic",
    encoder_config={"name": "thermometer", "num_bits": 4, "feature_wise": True},
    node_input_dim=6,
    layer_widths=[1024, 1000],
    num_classes=10,
    dataset="mnist",
    runtime={
        "temperature": 1.0,
        "init_fn": "xavier_uniform",
        "init_v_target": 0.25,
        "regularizers": {
            "entropy": {"weight": 0.001},
            "clarity": {"weight": 0.0001}
        }
    }
)

# 2. Build model
model = build_model(config)

# 3. Fit encoder on training data
train_loader = DataLoader(train_dataset, batch_size=128)
for batch in train_loader:
    data, _ = batch
    model.fit_encoder(data)
    break  # Fit on first batch

# 4. Setup training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# 5. Training loop
for epoch in range(num_epochs):
    model.train()
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(data)
        
        # Loss calculation
        loss = criterion(outputs, labels)
        reg_loss = model.get_regularization_loss()
        total_loss = loss + reg_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch}: Loss={total_loss.item():.4f}")

# 6. Save model
model.save_config("saved_models/my_model.yaml")
model.save_weights("saved_models/my_model.pth")
```

---

## Saving and Loading

### Save Model Configuration

```python
# Save config to YAML
model.save_config("models/my_model.yaml")
```

### Save Model Weights

```python
# Save weights
model.save_weights("models/my_model.pth")
```

### Load Model Weights

```python
# Load weights into existing model
model.load_weights("models/my_model.pth")
```

### Save as Pretrained Model

```python
# 1. Update config with pretrained info
config = model.get_config()
config.pretrained = True
config.pretrained_name = "my_pretrained_model"

# 2. Save to pretrained directory
model.save_config("difflut/models/pretrained/feedforward/my_pretrained_model.yaml")
model.save_weights("difflut/models/pretrained/feedforward/my_pretrained_model.pth")

# 3. Now it's available via build_model
new_model = build_model("my_pretrained_model", load_weights=True)
```

---

## Advanced Usage

### Custom Node Types

```python
config = ModelConfig(
    model_type="feedforward",
    node_type="dwn",  # Different node type
    runtime={
        "use_cuda": True  # Node-specific parameter
    },
    ...
)
```

### Residual Layers

```python
config = ModelConfig(
    model_type="feedforward",
    layer_type="residual",  # Residual connections
    ...
)
```

### Model Inspection

```python
# Count parameters
params = model.count_parameters()
print(f"Total: {params['total']:,}")
print(f"Trainable: {params['trainable']:,}")

# Get configuration
config = model.get_config()
print(config)

# Get layer topology (for SimpleFeedForward)
if hasattr(model, 'layers'):
    print(f"Number of layers: {len(model.layers)}")
    print(f"Layer widths: {config.layer_widths}")
```

### Check Config Compatibility

```python
# Check if two configs are compatible for weight sharing
config1 = ModelConfig(...)
config2 = ModelConfig(...)

compatible = config1.is_compatible_with_weights(config2)
print(f"Configs compatible: {compatible}")
```

---

## Migration from Old System

### Old LayeredFeedForward Style

```python
# OLD WAY
from experiments.models.layered_feedforward import LayeredFeedForward

config = {
    'encoder': {'name': 'thermometer', 'num_bits': 4},
    'layer_type': 'random',
    'node_type': 'probabilistic',
    'node_input_dim': 6,
    'hidden_sizes': [1024, 1000],
    'node_params': {
        'init_fn': 'xavier_uniform',
        'temperature': 1.0
    }
}
model = LayeredFeedForward(config, input_size=784, num_classes=10)
```

### New SimpleFeedForward Style

```python
# NEW WAY
from difflut.models import ModelConfig, SimpleFeedForward

config = ModelConfig(
    model_type="feedforward",
    layer_type="random",
    node_type="probabilistic",
    encoder_config={'name': 'thermometer', 'num_bits': 4},
    node_input_dim=6,
    layer_widths=[1024, 1000],
    num_classes=10,
    input_size=784,
    runtime={
        'init_fn': 'xavier_uniform',
        'temperature': 1.0
    }
)
model = SimpleFeedForward(config)
```

Or even simpler:

```python
# SIMPLEST WAY
from difflut.models import build_model

model = build_model("configs/my_model.yaml")
```

---

## Troubleshooting

### Encoder Not Fitted Error

```
RuntimeError: Encoder must be fitted before forward pass. Call fit_encoder() first.
```

**Solution:** Fit the encoder on training data before using the model:

```python
model.fit_encoder(train_data[:1000])  # Fit on sample of data
```

### Weight Loading Error

```
FileNotFoundError: Pretrained model 'xyz' not found
```

**Solution:** Check available models:

```python
from difflut.models import list_pretrained_models
print(list_pretrained_models())
```

### Config Compatibility Error

If loading weights fails with shape mismatch:

```python
# Check if configs are compatible
old_config = ModelConfig.from_yaml("old_model.yaml")
new_config = ModelConfig.from_yaml("new_model.yaml")

if not old_config.is_compatible_with_weights(new_config):
    print("Configs are incompatible - structural parameters differ")
```

---

## Summary

The new model system provides:

✅ **Clean Configuration** - Separate structural vs runtime parameters  
✅ **Pretrained Models** - Easy loading of pretrained weights  
✅ **Runtime Overrides** - Safe parameter changes without retraining  
✅ **Flexible Building** - Multiple ways to create models  
✅ **Better Organization** - Registry integration and discovery  
✅ **Backward Compatibility** - Works alongside existing code  

For more examples, see:
- `difflut/models/pretrained/README.md`
- `difflut/models/feedforward.py`
- Example notebooks (coming soon)
