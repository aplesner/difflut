"""
DiffLUT: Differentiable Look-Up Table Networks

Warning Control:
---------------
DiffLUT issues warnings to help you avoid common pitfalls:
- DeprecationWarning: For deprecated features
- UserWarning: For misuse or suboptimal configurations
- RuntimeWarning: For runtime anomalies (e.g., missing CUDA extensions)

To control warnings, use Python's warnings module:

# Suppress all DiffLUT warnings:
import warnings
warnings.filterwarnings('ignore', module='difflut')

# Suppress specific warning types:
warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut')
warnings.filterwarnings('ignore', category=UserWarning, module='difflut')

# Suppress warnings from specific modules:
warnings.filterwarnings('ignore', module='difflut.nodes.dwn_node')
warnings.filterwarnings('ignore', module='difflut.layers.learnable_layer')

# Show all warnings (useful during development):
warnings.filterwarnings('always', module='difflut')

# Treat warnings as errors (strict mode):
warnings.filterwarnings('error', module='difflut')
"""

from .encoder.base_encoder import BaseEncoder
from .layers.base_layer import BaseLUTLayer

# Import base classes
from .nodes.base_node import BaseNode

# Import registry first
from .registry import (
    REGISTRY,
    register_convolutional_layer,
    register_encoder,
    register_initializer,
    register_layer,
    register_node,
    register_regularizer,
)

__version__ = "1.4.2"

__all__ = [
    "REGISTRY",
    "register_node",
    "register_layer",
    "register_convolutional_layer",
    "register_encoder",
    "register_initializer",
    "register_regularizer",
    "BaseNode",
    "BaseLUTLayer",
    "BaseEncoder",
]
