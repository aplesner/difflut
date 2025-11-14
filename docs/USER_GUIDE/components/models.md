# Using DiffLUT Models

Learn how to load, configure, and use DiffLUT models for inference and training.

---

## Quick Start

### Load and Use a Pretrained Model

```python
from difflut.models import build_model

# Load a pretrained model (latest version)
model = build_model("mnist_large", load_weights=True)

# Or load a specific version explicitly
model = build_model("feedforward/mnist_large/v1", load_weights=True)

# Or with full path and version
model = build_model("feedforward/mnist_large/v2", load_weights=True)

# Use for inference
predictions = model(test_data)
```

### Build from YAML Config

```python
from difflut.models import build_model

# Build from configuration file
model = build_model("configs/my_model.yaml")

# Fit encoder and use
model.fit_encoder(train_data)
predictions = model(test_data)
```

### Override Runtime Parameters

```python
# Load model but adjust runtime behavior
model = build_model(
    "mnist_large",
    load_weights=True,
    overrides={
        "temperature": 0.5,
        "eval_mode": "sampling",
        "flip_probability": 0.01
    }
)
```

---

## Basic Usage Patterns

### Pattern 1: Inference with Pretrained Model

```python
from difflut.models import build_model

# Load pretrained model (already has encoder fitted and weights)
model = build_model("mnist_large", load_weights=True)

# Use for inference
model.eval()
with torch.no_grad():
    predictions = model(test_images)
```

### Pattern 2: Training from Scratch

```python
from difflut.models import build_model, ModelConfig

# Create configuration
config = ModelConfig(
    model_type="feedforward",
    layer_type="random",
    node_type="probabilistic",
    encoder_config={"name": "thermometer", "num_bits": 4},
    node_input_dim=6,
    layer_widths=[1024, 1000],
    num_classes=10
)

# Build model
model = build_model(config)

# Fit encoder on your data
model.fit_encoder(train_data)

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for batch, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, targets)
        
        # Add regularization
        reg_loss = model.get_regularization_loss()
        total_loss = loss + 0.001 * reg_loss
        
        total_loss.backward()
        optimizer.step()

# Evaluate
model.eval()
with torch.no_grad():
    accuracy = evaluate(model, test_loader)
```

### Pattern 3: Parameter Exploration

```python
# Compare model behavior with different runtime parameters
temperatures = [0.1, 0.5, 1.0, 2.0]
results = {}

for temp in temperatures:
    model = build_model(
        "mnist_large",
        load_weights=True,
        overrides={"temperature": temp}
    )
    accuracy = evaluate(model, test_loader)
    results[temp] = accuracy

print(results)
```

### Pattern 4: Ensemble Methods

```python
# Create ensemble of same model with different temperatures
models = [
    build_model("mnist_large", overrides={"temperature": 0.5}),
    build_model("mnist_large", overrides={"temperature": 1.0}),
    build_model("mnist_large", overrides={"temperature": 2.0}),
]

# Average predictions
with torch.no_grad():
    predictions = sum(m(test_data) for m in models) / len(models)
```

---

## Model Configuration

### Creating a Model Configuration

Configurations specify the model architecture and runtime parameters:

```python
from difflut.models import ModelConfig

config = ModelConfig(
    # Architecture parameters
    model_type="feedforward",          # Type of model
    layer_type="random",               # How layers route inputs
    node_type="probabilistic",         # Type of LUT nodes
    encoder_config={
        "name": "thermometer",
        "num_bits": 4,
        "feature_wise": True
    },
    node_input_dim=6,                  # Bits per node
    layer_widths=[1024, 1000],         # Hidden layer sizes
    num_classes=10,                    # Output classes
    input_size=784,                    # Input features (optional)
    dataset="mnist",                   # Dataset name (optional)
    
    # Runtime parameters (can be overridden)
    runtime={
        "temperature": 1.0,
        "eval_mode": "expectation",
        "flip_probability": 0.0
    },
    
    # Pretrained info
    pretrained=False,
    pretrained_name=None
)
```

### Configuration from YAML File

```yaml
# config.yaml
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

# Runtime parameters (can be overridden)
runtime:
  temperature: 1.0
  eval_mode: expectation
  flip_probability: 0.0

pretrained: false
```

Then load it:

```python
from difflut.models import build_model

model = build_model("config.yaml")
```

---

## Available Models

### Listing Available Models

```python
from difflut.models import list_pretrained_models

models = list_pretrained_models()
# Returns dict like:
# {
#     "feedforward": [
#         "mnist_large/v1",                 # versioned (all models now versioned)
#         "mnist_large/v2",
#         "cifar10_ffn_baseline/v1",
#         "cifar10_ffn_baseline/v2",
#         "mnist_small/v1",
#     ]
# }

# List all available models
for model_type, model_names in models.items():
    print(f"\n{model_type}:")
    for name in model_names:
        print(f"  - {name}")
```

### Getting Model Information

```python
from difflut.models import get_pretrained_model_info

info = get_pretrained_model_info("mnist_large")
print(info['model_type'])
print(info['layer_widths'])
print(info['has_weights'])
```

### Common Pretrained Models

Typical pretrained models follow this naming convention:
- `{dataset}_{architecture}_{size}`: e.g., `mnist_large`, `cifar10_small`
- Available variants: small, medium, large based on layer widths

---

## Runtime Parameters

Runtime parameters can be safely overridden without affecting model weights or structure.

### Common Runtime Parameters

```python
runtime_overrides = {
    # Bit flipping (test robustness)
    "flip_probability": 0.0,         # Probability to flip input bits
    
    # Gradient stabilization
    "grad_stabilization": "none",    # "none", "layerwise", "nodewise", "batch"
    
    # Regularization weights
    "regularizers": {
        "l1": 0.001,
    },
    
    # Training initialization
    "init_fn": "xavier_uniform",
    "init_v_target": 0.25
}

model = build_model(
    "mnist_large",
    overrides=runtime_overrides
)
```


---

## Working with Model Components

### Accessing Encoder

```python
# Get encoder for converting raw data
encoded_data = model.encode(raw_data)

# Or manually encode
from difflut.registry import REGISTRY

encoder_class = REGISTRY.get_encoder("thermometer")
encoder = encoder_class(num_bits=4, feature_wise=True)
encoder.fit(train_data)
encoded = encoder(test_data)
```

### Computing Regularization

```python
# Get total regularization loss
reg_loss = model.get_regularization_loss()

# Use in training
total_loss = criterion_loss + 0.001 * reg_loss
```

### Model Inspection

```python
# Count parameters
params = model.count_parameters()
print(f"Total: {params['total']:,}")
print(f"Trainable: {params['trainable']:,}")

# Get configuration
config = model.get_config()
print(config.layer_widths)
print(config.node_type)
```

---

## Training Models

### Complete Training Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from difflut.models import build_model, ModelConfig

# 1. Create and build model
config = ModelConfig(
    model_type="feedforward",
    layer_type="random",
    node_type="probabilistic",
    encoder_config={"name": "thermometer", "num_bits": 4},
    node_input_dim=6,
    layer_widths=[1024, 1000],
    num_classes=10,
    dataset="mnist"
)
model = build_model(config)

# 2. Fit encoder on training data
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
all_data = torch.cat([batch for batch, _ in train_loader])
model.fit_encoder(all_data)

# 3. Setup training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 4. Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Add regularization
        reg_loss = model.get_regularization_loss()
        total_loss_value = loss + 0.001 * reg_loss
        
        total_loss_value.backward()
        optimizer.step()
        total_loss += total_loss_value.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={total_loss / (batch_idx+1):.4f}")

# 5. Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)
        
        outputs = model(data)
        _, predicted = outputs.max(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

accuracy = 100 * correct / total
print(f"Final Accuracy: {accuracy:.2f}%")
```

### Key Training Tips

1. **Always fit encoder first** - Before forward pass: `model.fit_encoder(train_data)`
2. **Include regularization** - Add regularization loss to total: `total_loss = loss + 0.001 * reg_loss`
3. **Use appropriate learning rate** - Usually 0.001-0.01 for Adam
4. **Set model modes correctly** - `model.train()` and `model.eval()`
5. **Move to device** - `model.to(device)` and `data.to(device)`

---

## Saving and Loading Models

#### Save Model Configuration

```python
# Save architecture to YAML
model.save_config("my_model.yaml")
```

#### Save Model Weights

```python
# Save trained weights
model.save_weights("my_model.pth")
```

#### Load Model Weights

```python
# Load weights into existing model
model = build_model(config)
model.load_weights("my_model.pth")
```

#### Complete Save/Load Cycle

```python
# After training
model.save_config("trained_model.yaml")
model.save_weights("trained_model.pth")

# Later, reload
model = build_model("trained_model.yaml")
model.load_weights("trained_model.pth")

# Use for inference
predictions = model(test_data)
```

---

## Common Workflows

### Workflow 1: Quick Inference with Pretrained Model

```python
from difflut.models import build_model

# One line - load and use (automatically uses latest version)
model = build_model("mnist_large", load_weights=True)
predictions = model(test_data)

# Or explicitly specify version
model = build_model("feedforward/mnist_large/v2", load_weights=True)
```

### Workflow 2: Experiment with Different Parameters

```python
from difflut.models import build_model

# Load base model
model = build_model("mnist_large", load_weights=True)

# Test different configurations
configs_to_test = [
    {"temperature": 0.5},
    {"temperature": 1.0},
    {"temperature": 2.0},
    {"flip_probability": 0.05},
]

for config in configs_to_test:
    test_model = build_model("mnist_large", overrides=config)
    accuracy = evaluate(test_model, test_loader)
    print(f"Config {config}: {accuracy:.2%}")
```

### Workflow 3: Train New Model on Custom Data

```python
from difflut.models import build_model, ModelConfig

# Define architecture
config = ModelConfig(
    model_type="feedforward",
    layer_type="random",
    node_type="probabilistic",
    encoder_config={"name": "thermometer", "num_bits": 4},
    node_input_dim=6,
    layer_widths=[512, 256],
    num_classes=len(class_names),
)

# Build, train, save
model = build_model(config)
model.fit_encoder(train_data)

# Train (see complete example above)
# ...

# Save for later use
model.save_config("custom_model.yaml")
model.save_weights("custom_model.pth")
```

### Workflow 4: Robustness Testing

```python
from difflut.models import build_model

# Test model robustness to bit flips
flip_rates = [0.0, 0.01, 0.05, 0.1, 0.2]

for rate in flip_rates:
    model = build_model(
        "mnist_large",
        overrides={"flip_probability": rate}
    )
    accuracy = evaluate(model, test_loader)
    print(f"Flip rate {rate:.0%}: {accuracy:.2%}")
```

---


## Next Steps

### Learn More
- [Components Overview](./overview.md) - Learn about encoders, nodes, and layers
- [Registry & Pipelines](../registry_pipeline.md) - Advanced usage, building custom components
- [Export Guide](../export_guide.md) - Export models for deployment

### Create Custom Models
- See [Creating Custom Models](../../DEVELOPER_GUIDE/creating_components/creating_models.md)

---

