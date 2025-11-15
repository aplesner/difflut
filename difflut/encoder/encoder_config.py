"""
Configuration class for encoder parameters.

This module provides a typed, maintainable way to specify encoder configuration
following the same pattern as ModelConfig, NodeConfig, LayerConfig, and BlockConfig.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EncoderConfig:
    """
    Configuration for encoder parameters (type-safe alternative to dict).

    This class provides type-safe configuration for all encoder types with proper
    defaults and documentation. It can be converted to/from dictionaries for
    unified API compatibility.

    Common Parameters (all encoders):
        name: Name of encoder type (e.g., "thermometer", "binary", "onehot")
        num_bits: Number of bits in the encoded representation

    Encoder-Specific Parameters (stored in extra_params):
        ThermometerEncoder / GaussianThermometerEncoder / DistributiveThermometerEncoder:
            - No additional parameters (only num_bits, flatten)

        BinaryEncoder / GrayEncoder:
            - No additional parameters (only num_bits, flatten)

        OneHotEncoder:
            - No additional parameters (only num_bits, flatten)

        SignMagnitudeEncoder:
            - No additional parameters (only num_bits, flatten)

        LogarithmicEncoder:
            - base: Logarithm base for encoding (default: 2.0)

    Common Parameters (all encoders):
        flatten: If True, return 2D (batch_size, input_dim * num_bits);
                if False, return 3D (batch_size, input_dim, num_bits)

    Example:
        ```python
        # Simple thermometer encoder
        config = EncoderConfig(
            name="thermometer",
            num_bits=4,
            flatten=True
        )

        # Logarithmic encoder with custom base
        config = EncoderConfig(
            name="logarithmic",
            num_bits=8,
            base=2.0,
            flatten=True
        )

        # Use in model
        model = SimpleFeedForward(
            config=ModelConfig(
                model_type="feedforward",
                params={
                    "encoder_config": config.to_dict(),  # Convert to dict for ModelConfig
                    ...
                }
            )
        )
        ```
    """

    # ========================================================================
    # Common parameters (used by all encoders)
    # ========================================================================
    name: str
    """Encoder type name (e.g., "thermometer", "binary", "onehot", "logarithmic")"""

    num_bits: int = 4
    """Number of bits in the encoded representation"""

    flatten: bool = True
    """If True, return 2D tensor (batch_size, input_dim * num_bits); 
       if False, return 3D tensor (batch_size, input_dim, num_bits)"""

    # ========================================================================
    # Encoder-specific parameters (stored in extra_params)
    # ========================================================================
    extra_params: Dict[str, Any] = field(default_factory=dict)
    """Additional encoder-specific parameters stored here"""

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("encoder name must be provided")
        if self.num_bits <= 0:
            raise ValueError("num_bits must be positive")
        if not isinstance(self.flatten, bool):
            raise ValueError("flatten must be a boolean")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for use with encoders.

        Returns:
            Dictionary with all encoder parameters
        """
        result = {
            "name": self.name,
            "num_bits": self.num_bits,
            "flatten": self.flatten,
        }
        # Add extra parameters (encoder-specific)
        result.update(self.extra_params)
        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EncoderConfig":
        """
        Create EncoderConfig from dictionary.

        Args:
            data: Dictionary with encoder configuration

        Returns:
            EncoderConfig instance
        """
        # Extract known fields
        name = data.get("name")
        num_bits = data.get("num_bits", 4)
        flatten = data.get("flatten", True)

        # Everything else goes to extra_params
        extra_params = {
            k: v for k, v in data.items() if k not in ("name", "num_bits", "flatten")
        }

        return EncoderConfig(
            name=name,
            num_bits=num_bits,
            flatten=flatten,
            extra_params=extra_params,
        )

    def copy(self) -> "EncoderConfig":
        """Create a deep copy of this configuration."""
        return EncoderConfig(
            name=self.name,
            num_bits=self.num_bits,
            flatten=self.flatten,
            extra_params=self.extra_params.copy(),
        )

    def __repr__(self) -> str:
        """String representation showing all parameters."""
        params = [f"name='{self.name}'", f"num_bits={self.num_bits}"]
        if not self.flatten:
            params.append("flatten=False")
        if self.extra_params:
            extra_str = ", ".join(f"{k}={v}" for k, v in self.extra_params.items())
            params.append(extra_str)
        return f"EncoderConfig({', '.join(params)})"
