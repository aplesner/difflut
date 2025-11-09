# Registry & Pipeline Guide

Learn how to use DiffLUT's component registry for dynamic component discovery and configuration-driven model building.

---

## Table of Contents
1. [What is the Registry?](#what-is-the-registry)
2. [Listing Available Components](#listing-available-components)
3. [Dynamic Component Instantiation](#dynamic-component-instantiation)
4. [Configuration-Driven Model Building](#configuration-driven-model-building)
5. [Pipeline Construction Patterns](#pipeline-construction-patterns)
6. [Best Practices](#best-practices)

---

## What is the Registry?

The **component registry** is a central system that:
- Keeps track of all available components (encoders, nodes, layers, initializers, regularizers)
- Allows dynamic component instantiation by name
- Enables configuration-file-driven model building
- Supports easy component discovery

This is especially useful for:
- Hyperparameter tuning with different components
- Loading models from configuration files
- Building pipelines programmatically
- Research experiments with many configurations

---

## Listing Available Components

### Get All Registered Components

```python
from difflut.registry import REGISTRY

# List all available node types
print("Available nodes:")
print(REGISTRY.list_nodes())
# Output: ['linear_lut', 'polylut', 'neurallut', 'dwn', 'dwn_stable', 
#          'probabilistic', 'fourier', 'hybrid']

# List all available layer types
print("\nAvailable layers:")
print(REGISTRY.list_layers())
# Output: ['random', 'learnable']

# List all available encoders
print("\nAvailable encoders:")
print(REGISTRY.list_encoders())
# Output: ['thermometer', 'gaussian_thermometer', 'distributive_thermometer', 
#          'gray', 'onehot', 'binary', 'sign_magnitude', 'logarithmic']

# List all available initializers
print("\nAvailable initializers:")
print(REGISTRY.list_initializers())
# Output: ['kaiming_normal', 'kaiming_normal_init', 'kaiming_uniform', 
#          'kaiming_uniform_init', 'normal', 'normal_init', 'ones', 
#          'ones_init', 'uniform', 'uniform_init', 'xavier_normal', 
#          'xavier_normal_init', 'xavier_uniform', 'xavier_uniform_init', 
#          'zeros', 'zeros_init', ...]

# List all available regularizers
print("\nAvailable regularizers:")
print(REGISTRY.list_regularizers())
# Output: ['l', 'l1', 'l1_functional', 'l1_regularizer', 'l2', 'l2_functional', 
#          'l2_regularizer', 'l_regularizer', 'spectral', 'spectral_regularizer']
```

### Check if Component Exists

```python
from difflut.registry import REGISTRY

if 'linear_lut' in REGISTRY.list_nodes():
    print("✓ LinearLUT node is available")

if 'learnable' in REGISTRY.list_layers():
    print("✓ Learnable layer is available")

if 'kaiming_normal' in REGISTRY.list_initializers():
    print("✓ Kaiming normal initializer is available")

if 'l2' in REGISTRY.list_regularizers():
    print("✓ L2 regularizer is available")

# List all available components at once
all_components = REGISTRY.list_all()
print("\nRegistry contents:")
for category, items in all_components.items():
    print(f"  {category}: {len(items)} registered")
```

---

## Dynamic Component Instantiation

### Getting Component Classes

```python
from difflut.registry import REGISTRY
from difflut.nodes.node_config import NodeConfig
from difflut.layers.layer_config import LayerConfig

# Get class by name
NodeClass = REGISTRY.get_node('linear_lut')
LayerClass = REGISTRY.get_layer('random')
EncoderClass = REGISTRY.get_encoder('thermometer')

# Create instances with type-safe configuration
node_config = NodeConfig(input_dim=4, output_dim=1)
layer_config = LayerConfig(flip_probability=0.1, grad_stabilization='layerwise')

encoder = EncoderClass(num_bits=8)
layer = LayerClass(
    input_size=100,
    output_size=50,
    node_type=NodeClass,
    n=4,
    node_kwargs=node_config,
    layer_config=layer_config
)
```

### Building from Configuration Dictionary

```python
from difflut.registry import REGISTRY
from difflut.nodes.node_config import NodeConfig
from difflut.layers.layer_config import LayerConfig

# Configuration dictionary (from YAML/JSON)
config = {
    'node_type': 'polylut',
    'node_params': {
        'input_dim': 6,
        'output_dim': 1,
        'degree': 3  # Node-specific parameter
    },
    'layer_type': 'random',
    'layer_params': {
        'input_size': 512,
        'output_size': 256,
        'n': 6,
        'seed': 42
    },
    'layer_training': {
        'flip_probability': 0.1,
        'grad_stabilization': 'layerwise',
        'grad_target_std': 1.0
    }
}

# Build dynamically with typed configs
NodeClass = REGISTRY.get_node(config['node_type'])
LayerClass = REGISTRY.get_layer(config['layer_type'])

# Create NodeConfig (handles node-specific params via extra_params)
node_config = NodeConfig(
    input_dim=config['node_params']['input_dim'],
    output_dim=config['node_params']['output_dim'],
    extra_params={'degree': config['node_params'].get('degree')}
)

# Create LayerConfig
layer_config = LayerConfig(**config['layer_training'])

# Instantiate layer
layer = LayerClass(
    input_size=config['layer_params']['input_size'],
    output_size=config['layer_params']['output_size'],
    node_type=NodeClass,
    n=config['layer_params']['n'],
    node_kwargs=node_config,
    layer_config=layer_config,
    seed=config['layer_params'].get('seed', 42)
)
```

### Using Initializers and Regularizers from Registry

The registry provides centralized access to all initializer and regularizer functions. Always retrieve these from the registry and pass them through `NodeConfig`:

```python
from difflut.registry import REGISTRY
from difflut.nodes.node_config import NodeConfig

# Get initializer and regularizers from registry by name
init_fn = REGISTRY.get_initializer('kaiming_normal')
l2_reg = REGISTRY.get_regularizer('l2')
spectral_reg = REGISTRY.get_regularizer('spectral')

# Pass them to NodeConfig
node_config = NodeConfig(
    input_dim=6,
    output_dim=1,
    init_fn=init_fn,
    init_kwargs={'a': 0.0, 'mode': 'fan_in', 'nonlinearity': 'relu'},
    regularizers={
        'l2': l2_reg,
        'spectral': spectral_reg
    }
)

# Create node with full configuration
NodeClass = REGISTRY.get_node('linear_lut')
node = NodeClass(**node_config.to_dict())

# During training, compute regularization loss
reg_loss = node.regularization()
total_loss = task_loss + 0.01 * reg_loss
```

**Key Pattern:**
1. Use `REGISTRY.get_initializer(name)` to retrieve init functions
2. Use `REGISTRY.get_regularizer(name)` to retrieve regularizer functions
3. Pass retrieved functions to `NodeConfig` - do NOT pass function names as strings
4. Initialize the node with `NodeClass(**config.to_dict())`

**Why?**
- Registry maintains single source of truth for all components
- Type safety: functions are validated at retrieval time
- Easy to swap implementations or add new ones
- Configuration files can store names, code stores function objects

---

## Configuration-Driven Model Building

### YAML Configuration Example

Create a configuration file `model_config.yaml`:

```yaml
# model_config.yaml
encoder:
  type: thermometer
  params:
    num_bits: 8
    flatten: true

layers:
  - name: layer1
    type: random
    node_type: linear_lut
    input_size: 6272  # 784 * 8 from encoder
    output_size: 256
    n: 4
    seed: 42
    node_params:
      input_dim: 4
      output_dim: 1
    layer_training:
      flip_probability: 0.1
      grad_stabilization: layerwise
      grad_target_std: 1.0
    node_init:
      initializer: kaiming_normal
      init_kwargs:
        a: 0.0
        mode: fan_in
        nonlinearity: relu
  
  - name: layer2
    type: random
    node_type: dwn_stable
    input_size: 256
    output_size: 128
    n: 6
    seed: 43
    node_params:
      input_dim: 6
      output_dim: 1
      use_cuda: true
      gradient_scale: 1.0
    layer_training:
      flip_probability: 0.05
      grad_stabilization: batchwise
    node_regularizers:
      - l2
  
  - name: output
    type: random
    node_type: linear_lut
    input_size: 128
    output_size: 10
    n: 4
    seed: 44
    node_params:
      input_dim: 4
      output_dim: 1

groupsum:
  k: 10
  tau: 1.0
  use_randperm: false
```

### Loading Configuration and Building Model

```python
import yaml
import torch
import torch.nn as nn
from difflut.registry import REGISTRY
from difflut.nodes.node_config import NodeConfig
from difflut.layers.layer_config import LayerConfig
from difflut.utils.modules import GroupSum

# Load configuration
with open('model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Build encoder
encoder_config = config['encoder']
EncoderClass = REGISTRY.get_encoder(encoder_config['type'])
encoder = EncoderClass(**encoder_config['params'])

# Build layers with type-safe configuration
class ConfiguredLUTModel(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()
        self.encoder = encoder
        self.layers = nn.ModuleList()
        
        # Get node and layer classes from registry
        for layer_cfg in config['layers']:
            NodeClass = REGISTRY.get_node(layer_cfg['node_type'])
            LayerClass = REGISTRY.get_layer(layer_cfg['type'])
            
            # Build NodeConfig with registry-retrieved initializers/regularizers
            node_params = layer_cfg.get('node_params', {})
            
            # Get functions from registry if specified
            init_fn = None
            if 'init_fn_name' in node_params:
                init_fn = REGISTRY.get_initializer(node_params['init_fn_name'])
            
            regularizers = {}
            if 'regularizers' in node_params:
                for reg_name in node_params['regularizers']:
                    regularizers[reg_name] = REGISTRY.get_regularizer(reg_name)
            
            # Extract node-specific params
            extra_params = {k: v for k, v in node_params.items() 
                          if k not in ['init_fn_name', 'regularizers']}
            
            node_config = NodeConfig(
                input_dim=node_params.get('input_dim', 4),
                output_dim=node_params.get('output_dim', 1),
                init_fn=init_fn,
                init_kwargs=node_params.get('init_kwargs', {}),
                regularizers=regularizers if regularizers else None,
                extra_params=extra_params
            )
            
            # Build LayerConfig
            layer_training = layer_cfg.get('layer_training', {})
            layer_config = LayerConfig(**layer_training) if layer_training else None
            
            # Create layer
            layer = LayerClass(
                input_size=layer_cfg['input_size'],
                output_size=layer_cfg['output_size'],
                node_type=NodeClass,
                n=node_config.input_dim,
                node_kwargs=node_config,
                layer_config=layer_config,
                seed=layer_cfg.get('seed', 42)
            )
            self.layers.append(layer)
    
    def forward(self, x):
        x = x.flatten(1)  # Flatten input
        x = self.encoder(x)  # Encode
        for layer in self.layers:  # Process through layers
            x = layer(x)
        return x

# Create model from config
model = ConfiguredLUTModel(config, encoder)
print(f"Model created with {len(model.layers)} layers")
```

---

## Pipeline Construction Patterns

### Parameterized Pipeline Factory

```python
from difflut.registry import REGISTRY
from difflut.nodes.node_config import NodeConfig
from difflut.layers.layer_config import LayerConfig
from difflut.utils.modules import GroupSum
import torch
import torch.nn as nn

def build_lut_pipeline(
    input_size,
    hidden_sizes,
    num_classes,
    encoder_type='thermometer',
    encoder_params=None,
    layer_type='random',
    node_type='linear_lut',
    node_input_dim=4,
    node_params=None,
    layer_training_params=None,
    use_groupsum=True
):
    """
    Build a LUT network pipeline from parameters.
    
    Args:
        input_size: Input feature dimension (e.g., 784 for MNIST)
        hidden_sizes: List of hidden layer sizes [256, 128]
        num_classes: Number of output classes
        encoder_type: Name of encoder ('thermometer', 'gray', etc.)
        encoder_params: Dict of encoder parameters
        layer_type: Name of layer type ('random', 'learnable')
        node_type: Name of node type ('linear_lut', 'dwn_stable', etc.)
        node_input_dim: Number of inputs per node (n parameter)
        node_params: Dict of node-specific parameters (for extra_params)
        layer_training_params: Dict of LayerConfig parameters
        use_groupsum: Whether to add GroupSum at the end
    
    Returns:
        nn.Module: Built model
    """
    
    encoder_params = encoder_params or {'num_bits': 8, 'flatten': True}
    node_params = node_params or {}
    layer_training_params = layer_training_params or {}
    
    # Get classes from registry
    EncoderClass = REGISTRY.get_encoder(encoder_type)
    LayerClass = REGISTRY.get_layer(layer_type)
    NodeClass = REGISTRY.get_node(node_type)
    
    # Build encoder
    encoder = EncoderClass(**encoder_params)
    
    # Calculate encoded size
    num_bits = encoder_params.get('num_bits', 8)
    encoded_size = input_size * num_bits
    
    # Build layers
    layers = nn.ModuleList()
    layer_sizes = [encoded_size] + hidden_sizes + [num_classes]
    
    # Create shared configs
    layer_config = LayerConfig(**layer_training_params) if layer_training_params else None
    
    for i in range(len(layer_sizes) - 1):
        # Create NodeConfig for this layer
        # Retrieve initializers and regularizers from registry if specified in node_params
        init_fn = None
        if 'init_fn_name' in node_params:
            init_fn = REGISTRY.get_initializer(node_params['init_fn_name'])
        
        regularizers = {}
        if 'regularizer_names' in node_params:
            for reg_name in node_params['regularizer_names']:
                regularizers[reg_name] = REGISTRY.get_regularizer(reg_name)
        
        # Extract only extra params (remove init/reg names)
        extra_params = {k: v for k, v in node_params.items() 
                       if k not in ['init_fn_name', 'regularizer_names', 'init_kwargs']}
        
        node_config = NodeConfig(
            input_dim=node_input_dim,
            output_dim=1,
            init_fn=init_fn,
            init_kwargs=node_params.get('init_kwargs', {}),
            regularizers=regularizers if regularizers else None,
            extra_params=extra_params
        )
        
        layer = LayerClass(
            input_size=layer_sizes[i],
            output_size=layer_sizes[i + 1],
            node_type=NodeClass,
            n=node_input_dim,
            node_kwargs=node_config,
            layer_config=layer_config,
            seed=42 + i  # Different seed per layer
        )
        layers.append(layer)
    
    # Build GroupSum if requested
    groupsum = GroupSum(k=num_classes, tau=1.0) if use_groupsum else None
    
    # Wrap in model
    class LUTPipeline(nn.Module):
        def __init__(self, encoder, layers, groupsum):
            super().__init__()
            self.encoder = encoder
            self.layers = nn.ModuleList(layers)
            self.groupsum = groupsum
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.encoder(x)
            
            for i, layer in enumerate(self.layers[:-1]):
                x = torch.relu(layer(x))
            
            x = self.layers[-1](x)
            
            if self.groupsum is not None:
                x = self.groupsum(x)
            
            return x
    
    return LUTPipeline(encoder, layers, groupsum)

# Usage examples

# Basic model
model = build_lut_pipeline(
    input_size=784,
    hidden_sizes=[256, 128],
    num_classes=10,
    encoder_type='thermometer',
    encoder_params={'num_bits': 8},
    node_type='linear_lut',
    node_input_dim=4
)

# Model with robust training
model = build_lut_pipeline(
    input_size=784,
    hidden_sizes=[512, 256],
    num_classes=10,
    encoder_type='gray',
    encoder_params={'num_bits': 6},
    node_type='dwn_stable',
    node_input_dim=6,
    node_params={'use_cuda': True, 'gradient_scale': 1.0},
    layer_training_params={
        'flip_probability': 0.1,
        'grad_stabilization': 'layerwise',
        'grad_target_std': 1.0
    }
)
```

### Hyperparameter Search Using Registry

```python
from difflut.registry import REGISTRY
from difflut.nodes.node_config import NodeConfig
from difflut.layers.layer_config import LayerConfig
import itertools

# Define hyperparameter grid
param_grid = {
    'node_type': ['linear_lut', 'polylut', 'dwn_stable'],
    'layer_type': ['random', 'learnable'],
    'node_input_dim': [4, 6, 8],
    'hidden_size': [128, 256],
    'flip_probability': [0.0, 0.1],
    'grad_stabilization': ['none', 'layerwise']
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
    print(f"Testing config: {config}")
    
    # Build model using pipeline factory
    model = build_lut_pipeline(
        input_size=784,
        hidden_sizes=[config['hidden_size']],
        num_classes=10,
        encoder_type='thermometer',
        encoder_params={'num_bits': 8},
        layer_type=config['layer_type'],
        node_type=config['node_type'],
        node_input_dim=config['node_input_dim'],
        layer_training_params={
            'flip_probability': config['flip_probability'],
            'grad_stabilization': config['grad_stabilization']
        }
    )
    
    # Train and evaluate (pseudocode)
    # val_acc = train_and_evaluate(model, train_loader, val_loader)
    # results.append({'config': config, 'val_acc': val_acc})

# Find best configuration
# best_config = max(results, key=lambda x: x['val_acc'])
# print(f"Best config: {best_config['config']}")
# print(f"Best accuracy: {best_config['val_acc']:.4f}")
```

---

---

## Best Practices

1. **Store configurations with trained models** for reproducibility
   ```python
   import json
   
   # Save config with model
   torch.save({
       'model_state': model.state_dict(),
       'config': config_dict,
       'encoder_state': encoder.state_dict()
   }, 'model_checkpoint.pt')
   ```

2. **Use NodeConfig and LayerConfig** for type-safe parameter passing
   ```python
   # Type-safe (recommended)
   node_config = NodeConfig(input_dim=6, output_dim=1)
   layer_config = LayerConfig(flip_probability=0.1)
   
   # Avoid raw dictionaries
   # node_kwargs = {'input_dim': 6, 'output_dim': 1}  # Less safe
   ```

3. **Leverage registry for ablation studies** - vary one component at a time
   ```python
   for node_type in ['linear_lut', 'polylut', 'dwn_stable']:
       model = build_model(node_type=node_type, ...)
       results[node_type] = evaluate(model)
   ```

4. **Document node-specific parameters** in configuration files
   ```yaml
   node_params:
     input_dim: 6
     output_dim: 1
     # DWN-specific parameters
     use_cuda: true
     gradient_scale: 1.0
   ```

5. **Version your configurations** alongside model checkpoints
   ```python
   config['version'] = '1.0'
   config['timestamp'] = datetime.now().isoformat()
   ```

6. **Validate configurations before building**
   ```python
   def validate_config(config):
       assert 'encoder' in config
       assert 'layers' in config
       for layer in config['layers']:
           assert 'input_size' in layer
           assert 'output_size' in layer
           assert 'node_type' in layer
       return True
   
   validate_config(config)
   model = build_from_config(config)
   ```

7. **Use consistent random seeds** for reproducibility
   ```python
   config = {
       'seed': 42,
       'layers': [
           {'seed': 42, ...},
           {'seed': 43, ...},  # Different per layer
           {'seed': 44, ...},
       ]
   }
   ```

---


## Next Steps

- **[Components Guide](components.md)** - Detailed reference of all components
- **[Quick Start](../QUICK_START.md)** - Build your first model
- **[Creating Components](../DEVELOPER_GUIDE/creating_components.md)** - Implement custom components with registry decorators
