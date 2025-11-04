# Registry & Pipeline Guide

Learn how to use DiffLUT's component registry for dynamic component discovery and configuration-driven model building.

## What is the Registry?

The **component registry** is a central system that:
- Keeps track of all available components (encoders, nodes, layers)
- Allows dynamic component instantiation by name
- Enables configuration-file-driven model building
- Supports easy component discovery

This is especially useful for:
- Hyperparameter tuning with different components
- Loading models from configuration files
- Building pipelines programmatically
- Research experiments with many configurations

## Listing Available Components

### Get All Registered Components

```python
from difflut.registry import (
    get_registered_nodes,
    get_registered_layers,
    get_registered_encoders
)

# List all available node types
print("Available nodes:")
print(get_registered_nodes())
# Output: ['linear_lut', 'polylut', 'neurallut', 'dwn', 'dwn_stable', 'probabilistic', 'fourier', 'hybrid']

# List all available layer types
print("\nAvailable layers:")
print(get_registered_layers())
# Output: ['random', 'learnable']

# List all available encoders
print("\nAvailable encoders:")
print(get_registered_encoders())
# Output: ['thermometer', 'gaussian_thermometer', 'distributive_thermometer', 'gray', 'onehot', 'binary', 'sign_magnitude', 'logarithmic']
```

### Check if Component Exists

```python
from difflut.registry import is_registered_node, is_registered_layer

if is_registered_node('linear_lut'):
    print("✓ LinearLUT node is available")

if is_registered_layer('learnable'):
    print("✓ Learnable layer is available")
```

## Dynamic Component Instantiation

### Getting Component Classes

```python
from difflut.registry import (
    get_node_class,
    get_layer_class,
    get_encoder_class
)
from difflut.nodes.node_config import NodeConfig

# Get class by name
NodeClass = get_node_class('linear_lut')
LayerClass = get_layer_class('random')
EncoderClass = get_encoder_class('thermometer')

# Create instances with NodeConfig for type-safe parameters
config = NodeConfig(input_dim=4, output_dim=1)
encoder = EncoderClass(num_bits=8)
layer = LayerClass(
    input_size=100,
    output_size=50,
    node_type=NodeClass,
    n=4,
    node_kwargs=config
)
```

### Building from Configuration

```python
from difflut.registry import get_node_class, get_layer_class
from difflut.nodes.node_config import NodeConfig

# Configuration dictionary (traditional approach)
config = {
    'node_type': 'poly_lut',
    'node_params': {
        'input_dim': 6,
        'output_dim': 1,
        'extra_params': {'degree': 3}
    },
    'layer_type': 'random',
    'layer_params': {
        'input_size': 512,
        'output_size': 256,
        'n': 6
    }
}

# Build dynamically with NodeConfig
NodeClass = get_node_class(config['node_type'])
LayerClass = get_layer_class(config['layer_type'])

# Create type-safe node configuration
node_config = NodeConfig(
    input_dim=config['node_params']['input_dim'],
    output_dim=config['node_params']['output_dim'],
    extra_params=config['node_params'].get('extra_params', {})
)

layer = LayerClass(
    input_size=config['layer_params']['input_size'],
    output_size=config['layer_params']['output_size'],
    node_type=NodeClass,
    n=config['layer_params']['n'],
    node_kwargs=node_config
)
```

**Note:** NodeConfig provides type-safe parameter passing while maintaining backward compatibility with dict-based APIs through the `to_dict()` method.

## Configuration-Driven Model Building

### YAML Configuration Example

Create a configuration file `model_config.yaml`:

```yaml
# model_config.yaml
encoder:
  type: thermometer
  params:
    num_bits: 8
    feature_wise: true

layers:
  - name: layer1
    type: random
    node_type: linear_lut
    input_size: 784
    output_size: 256
    n: 4
    flip_probability: 0.01
    grad_stabilization: layerwise
    grad_target_std: 1.0
    node_params:
      input_dim: 4
      output_dim: 1
  
  - name: layer2
    type: random
    node_type: poly_lut
    input_size: 256
    output_size: 128
    n: 6
    flip_probability: 0.01
    grad_stabilization: layerwise
    node_params:
      input_dim: 6
      output_dim: 1
      extra_params:
        degree: 3
  
  - name: output
    type: random
    node_type: linear_lut
    input_size: 128
    output_size: 10
    n: 4
    node_params:
      input_dim: 4
      output_dim: 1
```

### Loading Configuration and Building Model

```python
import yaml
import torch
import torch.nn as nn
from difflut.registry import (
    get_encoder_class,
    get_layer_class,
    get_node_class
)
from difflut.nodes.node_config import NodeConfig

# Load configuration
with open('model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Build encoder
encoder_config = config['encoder']
EncoderClass = get_encoder_class(encoder_config['type'])
encoder = EncoderClass(**encoder_config['params'])

# Build layers
class ConfiguredLUTModel(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()
        self.encoder = encoder
        self.layers = nn.ModuleList()
        
        for layer_config in config['layers']:
            NodeClass = get_node_class(layer_config['node_type'])
            LayerClass = get_layer_class(layer_config['type'])
            
            # Create type-safe node configuration from YAML
            node_params = layer_config['node_params']
            node_config = NodeConfig(
                input_dim=node_params['input_dim'],
                output_dim=node_params['output_dim'],
                extra_params=node_params.get('extra_params', {})
            )
            
            # Extract layer parameters (optional, with defaults)
            layer_kwargs = {
                'input_size': layer_config['input_size'],
                'output_size': layer_config['output_size'],
                'node_type': NodeClass,
                'n': layer_config['n'],
                'node_kwargs': node_config
            }
            
            # Add optional layer parameters if present in config
            if 'flip_probability' in layer_config:
                layer_kwargs['flip_probability'] = layer_config['flip_probability']
            if 'grad_stabilization' in layer_config:
                layer_kwargs['grad_stabilization'] = layer_config['grad_stabilization']
            if 'grad_target_std' in layer_config:
                layer_kwargs['grad_target_std'] = layer_config['grad_target_std']
            
            layer = LayerClass(**layer_kwargs)
            self.layers.append(layer)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Add ReLU between layers (except output)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        
        return x

# Create model from config
model = ConfiguredLUTModel(config, encoder)
```

## Pipeline Construction Patterns

### Parameterized Pipeline Factory

```python
from difflut.registry import get_node_class, get_layer_class, get_encoder_class
import torch.nn as nn

def build_lut_pipeline(
    input_size,
    hidden_sizes,
    output_size,
    encoder_type='thermometer',
    encoder_params=None,
    layer_type='random',
    node_type='linear_lut',
    node_params=None
):
    """
    Build a LUT network pipeline from parameters.
    
    Args:
        input_size: Input feature dimension
        hidden_sizes: List of hidden layer sizes
        output_size: Output dimension
        encoder_type: Type of encoder ('thermometer', 'gray', etc.)
        encoder_params: Dict of encoder parameters
        layer_type: Type of layer ('random', 'learnable', etc.)
        node_type: Type of node ('linear_lut', 'polylut', etc.)
        node_params: Dict of node parameters
    
    Returns:
        nn.Module: Built model
    """
    
    encoder_params = encoder_params or {'num_bits': 8}
    node_params = node_params or {'input_dim': [4], 'output_dim': [1]}
    
    # Get classes
    EncoderClass = get_encoder_class(encoder_type)
    LayerClass = get_layer_class(layer_type)
    NodeClass = get_node_class(node_type)
    
    # Build encoder
    encoder = EncoderClass(**encoder_params)
    
    # Build layers
    layers = nn.ModuleList()
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    
    for i in range(len(layer_sizes) - 1):
        layer = LayerClass(
            input_size=layer_sizes[i],
            output_size=layer_sizes[i + 1],
            node_type=NodeClass,
            n=node_params.get('input_dim', [4])[0],
            node_kwargs=node_params
        )
        layers.append(layer)
    
    # Wrap in model
    class LUTPipeline(nn.Module):
        def __init__(self, encoder, layers):
            super().__init__()
            self.encoder = encoder
            self.layers = layers
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.encoder(x)
            
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = torch.relu(x)
            
            return x
    
    return LUTPipeline(encoder, layers)

# Usage
model = build_lut_pipeline(
    input_size=784,
    hidden_sizes=[256, 128],
    output_size=10,
    encoder_type='thermometer',
    encoder_params={'num_bits': 8},
    layer_type='random',
    node_type='linear_lut',
    node_params={'input_dim': [4], 'output_dim': [1]}
)
```

### Hyperparameter Search Using Registry

```python
from difflut.registry import get_node_class, get_layer_class
import itertools
import torch.nn as nn

# Define hyperparameter grid
param_grid = {
    'node_type': ['linear_lut', 'polylut', 'neurallut'],
    'layer_type': ['random', 'learnable'],
    'n_inputs': [4, 6, 8],
    'hidden_size': [128, 256]
}

# Generate configurations
def config_generator(param_grid):
    """Generate all parameter combinations."""
    keys = param_grid.keys()
    for values in itertools.product(*param_grid.values()):
        yield dict(zip(keys, values))

# Build and test models
results = []
for config in config_generator(param_grid):
    # Get component classes
    NodeClass = get_node_class(config['node_type'])
    LayerClass = get_layer_class(config['layer_type'])
    
    # Create model
    model = build_simple_model(
        NodeClass,
        LayerClass,
        n=config['n_inputs'],
        hidden_size=config['hidden_size']
    )
    
    # Train and evaluate (pseudocode)
    # val_acc = train_and_evaluate(model, train_loader, val_loader)
    
    # results.append({
    #     'config': config,
    #     'val_acc': val_acc
    # })

# Find best configuration
# best_config = max(results, key=lambda x: x['val_acc'])
# print(f"Best config: {best_config['config']}")
```

## Advanced Registry Usage

### Registering Custom Components

To register your own components, see [Creating Components](../DEVELOPER_GUIDE/creating_components.md).

### Querying Component Metadata

```python
from difflut.registry import get_node_class

# Get class information
NodeClass = get_node_class('linear_lut')

# Check docstring
print(NodeClass.__doc__)

# Check available parameters
import inspect
sig = inspect.signature(NodeClass.__init__)
print(f"Parameters: {sig.parameters.keys()}")
```

## Example: Complete Pipeline

```python
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from difflut.registry import get_encoder_class, get_layer_class, get_node_class

# 1. Load configuration
with open('experiment_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 2. Build model from config
encoder_cls = get_encoder_class(config['encoder']['type'])
encoder = encoder_cls(**config['encoder']['params'])

layers = []
for layer_cfg in config['layers']:
    node_cls = get_node_class(layer_cfg['node_type'])
    layer_cls = get_layer_class(layer_cfg['type'])
    layer = layer_cls(
        input_size=layer_cfg['input_size'],
        output_size=layer_cfg['output_size'],
        node_type=node_cls,
        n=layer_cfg['n'],
        node_kwargs=layer_cfg['node_params']
    )
    layers.append(layer)

# 3. Define model
class Model(nn.Module):
    def __init__(self, encoder, layers):
        super().__init__()
        self.encoder = encoder
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x

model = Model(encoder, layers)

# 4. Train
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ... training loop ...

# 5. Save configuration for reproducibility
import json
with open('trained_model_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# 6. Later: Load and recreate model with same configuration
with open('trained_model_config.json', 'r') as f:
    loaded_config = json.load(f)

# Use same registry system to recreate model
# ...
```

## Best Practices

1. **Store configurations with trained models** for reproducibility
2. **Use parameter validation** before passing to registry
3. **Leverage registry for ablation studies** - vary one component at a time
4. **Document non-standard parameters** in configuration files
5. **Version your configurations** alongside model checkpoints

## Next Steps

- [Components Guide](components.md) - Detailed reference of all components
- [Quick Start](../QUICK_START.md) - Run first example
- [Developer Guide](../DEVELOPER_GUIDE.md) - Create custom components with registration
