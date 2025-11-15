"""
Encoder module for difflut.
Provides various encoding schemes for transforming continuous values to binary representations.
"""

from .advanced_encoders import (
    BinaryEncoder,
    GrayEncoder,
    LogarithmicEncoder,
    OneHotEncoder,
    SignMagnitudeEncoder,
)
from .base_encoder import BaseEncoder
from .encoder_config import EncoderConfig
from .thermometer import (
    DistributiveThermometerEncoder,
    GaussianThermometerEncoder,
    ThermometerEncoder,
)

__all__ = [
    "EncoderConfig",
    "BaseEncoder",
    "ThermometerEncoder",
    "GaussianThermometerEncoder",
    "DistributiveThermometerEncoder",
    "GrayEncoder",
    "OneHotEncoder",
    "BinaryEncoder",
    "SignMagnitudeEncoder",
    "LogarithmicEncoder",
]
