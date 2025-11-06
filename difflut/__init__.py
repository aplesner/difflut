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

# Import registry first
from .registry import (
    REGISTRY,
    register_node,
    register_layer,
    register_encoder,
    register_initializer,
    register_regularizer,
)

# Import base classes
from .nodes.base_node import BaseNode
from .layers.base_layer import BaseLUTLayer
from .encoder.base_encoder import BaseEncoder

# Import all implementations to trigger registration
from .nodes import (
    DWNNode,
    LinearLUTNode,
    NeuralLUTNode,
    PolyLUTNode,
    ProbabilisticNode,
    HybridNode,
    FourierNode,
    DWNStableNode,
)

from .layers import (
    RandomLayer,
    LearnableLayer,
)

from .encoder import (
    ThermometerEncoder,
    GaussianThermometerEncoder,
    DistributiveThermometerEncoder,
    GrayEncoder,
    OneHotEncoder,
    BinaryEncoder,
    SignMagnitudeEncoder,
    LogarithmicEncoder,
)

from .utils import (
    GroupSum,
    # Warning utilities
    DiffLUTWarning,
    PerformanceWarning,
    ConfigurationWarning,
    configure_warnings,
    enable_all_warnings,
    disable_all_warnings,
    disable_cuda_warnings,
    enable_strict_mode,
    suppress_warnings,
)

__version__ = "1.0.10"

__all__ = [
    'REGISTRY',
    'register_node',
    'register_layer',
    'register_encoder',
    'register_initializer',
    'register_regularizer',
    'BaseNode',
    'BaseLUTLayer',
    'BaseEncoder',
    
    # Nodes
    'DWNNode',
    'LinearLUTNode',
    'NeuralLUTNode',
    'PolyLUTNode',
    'ProbabilisticNode',
    'HybridNode',
    'FourierNode',
    'DWNStableNode',
    
    # Layers
    'RandomLayer',
    'LearnableLayer',
    
    # Encoders
    'ThermometerEncoder',
    'GaussianThermometerEncoder',
    'DistributiveThermometerEncoder',
    'GrayEncoder',
    'OneHotEncoder',
    'BinaryEncoder',
    'SignMagnitudeEncoder',
    'LogarithmicEncoder',

    # Utils
    'GroupSum',
    
    # Warning utilities
    'DiffLUTWarning',
    'PerformanceWarning',
    'ConfigurationWarning',
    'configure_warnings',
    'enable_all_warnings',
    'disable_all_warnings',
    'disable_cuda_warnings',
    'enable_strict_mode',
    'suppress_warnings',
]
