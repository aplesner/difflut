"""
Encoder module for difflut.
Provides various encoding schemes for transforming continuous values to binary representations.
"""

from .base_encoder import BaseEncoder
from .thermometer import (
    ThermometerEncoder,
    GaussianThermometerEncoder,
    DistributiveThermometerEncoder,

)
from .advanced_encoders import (
    GrayEncoder,
    OneHotEncoder,
    BinaryEncoder,
    SignMagnitudeEncoder,
    LogarithmicEncoder,
)

__all__ = [
    'BaseEncoder',
    'ThermometerEncoder',
    'GaussianThermometerEncoder',
    'DistributiveThermometerEncoder',
    'GrayEncoder',
    'OneHotEncoder',
    'BinaryEncoder',
    'SignMagnitudeEncoder',
    'LogarithmicEncoder',
]
