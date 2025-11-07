# User Guide

Welcome to the DiffLUT User Guide! This guide covers all the features you need to build and train LUT neural networks.

## Overview

DiffLUT is built around three core concepts:

1. **Encoders**: Transform continuous inputs into discrete representations suitable for LUT indexing

2. **Layers**: Connect inputs to nodes with specific connectivity patterns
3. **Nodes**: Individual LUT units that perform computation
3.1. **Initalizers**
3.2. **Regularizers**

Together, these form complete differentiable LUT networks.

## Quick Links

- **[Quick Start](QUICK_START.md)** - Get running in 5 minutes
- **[Components Guide](USER_GUIDE/components.md)** - Deep dive into encoders, nodes, and layers
- **[Registry & Pipelines](USER_GUIDE/registry_pipeline.md)** - Component discovery and configuration

## Architecture Overview

```
Input Data
    ↓
[Encoder] → Discretize continuous values → Binary/categorical indices
    ↓
[Layer] → Connect inputs to LUT nodes
    ↓
[Nodes] → Compute LUT values → Differentiable forward pass
    ↓
Output/Next Layer
```

## Component Categories

### Encoders
Transform continuous inputs into discrete representations. See [Components Guide](USER_GUIDE/components.md#encoders).

### Nodes
Define computation at individual LUT units. See [Components Guide](USER_GUIDE/components.md#nodes).

### Layers
Connect inputs to nodes with configurable connectivity patterns. See [Components Guide](USER_GUIDE/components.md#layers).


## Component Registry

DiffLUT's **component registry** provides a unified way to discover and instantiate components by name. This is especially useful when loading configurations from files.

See [Registry & Pipelines Guide](USER_GUIDE/registry_pipeline.md) for:
- Listing available components
- Dynamic component instantiation
- Configuration-driven model building
- Pipeline construction patterns

## Next Steps

1. **[Quick Start](QUICK_START.md)** - Build your first network
2. **[Components Guide](USER_GUIDE/components.md)** - Comprehensive component reference
3. **[Registry & Pipelines](USER_GUIDE/registry_pipeline.md)** - Advanced configuration patterns
4. **[FPGA Export](USER_GUIDE/fpga_export.md)** - How to run on an FPGA
5. **Examples** - Check `examples/` directory for full training examples

## Common Questions

**Q: Do I need to fit encoders?**
A: Yes, encoders learn value ranges from training data. Always call `encoder.fit(train_data)` before using.

**Q: Which node type should I use?**
A: Start with `LinearLUTNode` - it's fastest and often sufficient. Experiment with others if needed.

**Q: Can I mix different node types?**
A: Yes! Create different layers with different node types and stack them.

**Q: Is GPU support available?**
A: Yes! Fourier, Hybrid, and Gradient-Stabilized nodes have CUDA implementations for GPU acceleration.

**Q: How do I export models for FPGA?**
A: See **[FPGA Export](USER_GUIDE/fpga_export.md)**


