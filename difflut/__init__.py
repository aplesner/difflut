"""
DiffLUT: Differentiable Look-Up Table Networks
"""

# Import registry first
from .registry import REGISTRY, register_node, register_layer, register_encoder

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
    UnboundProbabilisticNode,
    HybridNode,
    FourierNode,
    FourierHermitianNode,
)

from .layers import (
    RandomLayer,
    LearnableLayer,
    GroupedLayer,
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

__version__ = "1.0.10"

__all__ = [
    # Registry
    'REGISTRY',
    'register_node',
    'register_layer',
    'register_encoder',
    
    # Base classes
    'BaseNode',
    'BaseLUTLayer',
    'BaseEncoder',
    
    # Nodes
    'DWNNode',
    'LinearLUTNode',
    'NeuralLUTNode',
    'PolyLUTNode',
    'ProbabilisticNode',
    'UnboundProbabilisticNode',
    'HybridNode',
    'FourierNode',
    'FourierHermitianNode',
    
    # Layers
    'RandomLayer',
    'LearnableLayer',
    'GroupedLayer',
    
    # Encoders
    'ThermometerEncoder',
    'GaussianThermometerEncoder',
    'DistributiveThermometerEncoder',
    'GrayEncoder',
    'OneHotEncoder',
    'BinaryEncoder',
    'SignMagnitudeEncoder',
    'LogarithmicEncoder',
]
