# README for Pretrained Models

This directory contains pretrained model configurations and weights for DiffLUT models.

## Directory Structure

```
pretrained/
├── feedforward/
│   ├── mnist_large.yaml         # Large MNIST model config
│   ├── mnist_large.pth          # Weights (not included in git)
│   ├── mnist_small.yaml         # Small MNIST model config
│   └── cifar10_large.yaml       # CIFAR-10 model config
└── convnet/
    └── ...
```

## Using Pretrained Models

### Load a pretrained model with weights

```python
from difflut.models import build_model

# Load model with pretrained weights
model = build_model("mnist_large", load_weights=True)

# Use the model for inference
output = model(input_data)
```

### Load architecture without weights (for training)

```python
# Load just the architecture, train from scratch
model = build_model("mnist_large", load_weights=False)
```

### Override runtime parameters

```python
# Load model but change temperature and eval mode
model = build_model(
    "mnist_large",
    load_weights=True,
    overrides={
        "temperature": 0.5,
        "eval_mode": "sampling"
    }
)
```

## Configuration Format

Each pretrained model has a YAML config file with:

### Structural Parameters (Cannot be overridden with pretrained weights)
- `model_type`: Type of model (feedforward, convnet, etc.)
- `layer_type`: Type of layers (random, residual, etc.)
- `node_type`: Type of nodes (probabilistic, dwn, etc.)
- `encoder_config`: Encoder configuration
- `node_input_dim`: Input dimension for nodes
- `layer_widths`: Width of each layer
- `num_classes`: Number of output classes

### Runtime Parameters (Safe to override)
- `temperature`: Temperature for probabilistic nodes
- `eval_mode`: Evaluation mode (expectation, sampling)
- `flip_probability`: Bit flip probability for testing
- `dropout`: Dropout rate
- And other runtime parameters in the `runtime` section

## Creating a New Pretrained Model

1. Train your model and save config + weights:

```python
from difflut.models import SimpleFeedForward, ModelConfig

# Create config
config = ModelConfig(
    model_type="feedforward",
    layer_type="random",
    node_type="probabilistic",
    encoder_config={"name": "thermometer", "num_bits": 4},
    node_input_dim=6,
    layer_widths=[1024, 1000],
    num_classes=10,
    dataset="mnist",
    pretrained=True,
    pretrained_name="my_model"
)

# Train model...
model = SimpleFeedForward(config)
model.fit_encoder(train_data)
# ... train ...

# Save config and weights
model.save_config("pretrained/feedforward/my_model.yaml")
model.save_weights("pretrained/feedforward/my_model.pth")
```

2. Update the config YAML to include:
   - Metadata (accuracy, training details)
   - Default runtime parameters
   - pretrained: true
   - pretrained_name: "my_model"

3. The model is now available via:

```python
model = build_model("my_model", load_weights=True)
```

## Weight Files

Weight files (`.pth`) are stored separately and can be:
- Downloaded from a model zoo
- Generated during training
- Stored in Git LFS for version control

Weight files are optional - you can use just the config to recreate the architecture.
