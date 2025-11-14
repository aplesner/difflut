"""
DiffLUT Models Module

Provides a unified interface for creating and managing DiffLUT models.

Main Components:
- ModelConfig: Configuration dataclass for model parameters
- BaseLUTModel: Base class for all DiffLUT models
- SimpleFeedForward: Simple feedforward network implementation
- build_model: Factory function for creating models
- load_pretrained: Convenience function for loading pretrained models

Usage Examples:
    # Load a pretrained model
    >>> from difflut.models import build_model
    >>> model = build_model("mnist_large", load_weights=True)
    
    # Build from config file
    >>> model = build_model("configs/my_model.yaml")
    
    # Build with runtime overrides
    >>> model = build_model("mnist_large", overrides={"temperature": 0.5})
    
    # Create custom model from config
    >>> from difflut.models import ModelConfig, SimpleFeedForward
    >>> config = ModelConfig(
    ...     model_type="feedforward",
    ...     layer_type="random",
    ...     node_type="probabilistic",
    ...     encoder_config={"name": "thermometer", "num_bits": 4},
    ...     node_input_dim=6,
    ...     layer_widths=[1024, 1000],
    ...     num_classes=10,
    ... )
    >>> model = SimpleFeedForward(config)
"""

# Core components
from .model_config import ModelConfig
from .base_model import BaseLUTModel

# Model implementations
from .feedforward import SimpleFeedForward

# Factory functions
from .factory import (
    build_model,
    build_model_for_experiment,
    load_pretrained,
    list_pretrained_models,
    get_pretrained_model_info,
)

__all__ = [
    # Core
    "ModelConfig",
    "BaseLUTModel",
    
    # Models
    "SimpleFeedForward",
    
    # Factory
    "build_model",
    "build_model_for_experiment",
    "load_pretrained",
    "list_pretrained_models",
    "get_pretrained_model_info",
]
