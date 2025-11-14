"""
Model configuration dataclass for DiffLUT models.

Provides a flexible configuration system that stores model-specific parameters.
Different models may have completely different architectures and parameters,
and this config adapts to each.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


@dataclass
class ModelConfig:
    """
    Flexible configuration for DiffLUT models (future-proof design).

    This config is model-agnostic and stores only the parameters relevant to
    each specific model. Different models may have completely different
    architectures (feedforward, convolutional, segmentation, regression, etc.)
    and this config adapts to each.

    Structural Parameters (must match pretrained weights):
        - model_type: Type of model class to instantiate (e.g., "feedforward", "convolutional")
        - All other fields store model-specific architectural parameters
        - Each model defines which of its parameters are structural vs runtime

    Runtime Parameters:
        - Stored in the 'runtime' dict
        - Each model type defines which of its parameters can be overridden at runtime
        - Examples: temperature, eval_mode, dropout, flip_probability, etc.

    Design Philosophy:
        - Store only parameters actually used by the model
        - No null/unused fields
        - Model classes define what is structural vs runtime, not this config
        - Support heterogeneous architectures (some with encoders, some without;
          some for classification, some for regression/segmentation, etc.)
    """

    # ==================== Core Parameters ====================
    # Only these two are always required

    model_type: str  # Class name to instantiate (e.g., "feedforward", "convolutional")
    seed: int = 42

    # ==================== Model-Specific Structural Parameters ====================
    # Store only parameters relevant to this specific model.
    # Different models may have completely different sets of parameters.
    # Each model's documentation specifies which parameters it requires.

    # Example for feedforward: layer_type, node_type, encoder_config, node_input_dim, layer_widths, num_classes
    # Example for convolutional: layer_type, node_type, encoder_config, node_input_dim, conv_layer_widths, num_classes
    # Example for regression: layer_type, node_type, encoder_config, node_input_dim, layer_widths, output_dim
    # Example for segmentation: layer_type, node_type, encoder_config, node_input_dim, conv_layer_widths

    # Flexible storage: any additional parameters go here
    params: Dict[str, Any] = field(default_factory=dict)

    # ==================== Runtime Parameters ====================
    # These can be safely overridden at runtime without affecting model structure.
    # Each model type defines which of its parameters are runtime-safe.

    runtime: Dict[str, Any] = field(default_factory=dict)

    # ==================== Pretrained Information ====================

    pretrained: bool = False
    pretrained_name: Optional[str] = None

    # ==================== Serialization ====================

    @staticmethod
    def from_yaml(path: str) -> "ModelConfig":
        """
        Load configuration from YAML file.

        The YAML can contain any parameters relevant to the model type.
        All non-reserved fields are stored in the 'params' dict.

        Args:
            path: Path to YAML configuration file

        Returns:
            ModelConfig instance

        Example YAML (feedforward):
            model_type: feedforward
            seed: 42
            layer_type: random
            node_type: probabilistic
            encoder_config:
              name: thermometer
              num_bits: 4
            node_input_dim: 6
            layer_widths: [1024, 1000]
            num_classes: 10
            runtime:
              temperature: 1.0
              eval_mode: expectation

        Example YAML (convolutional):
            model_type: convolutional
            seed: 42
            layer_type: random
            node_type: probabilistic
            encoder_config:
              name: thermometer
              num_bits: 4
            node_input_dim: 6
            conv_layer_widths: [32, 64, 128]
            num_classes: 10
            runtime:
              temperature: 1.0
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Ensure model_type is present
        if "model_type" not in data:
            raise ValueError("Missing required field in config: model_type")

        # Extract reserved fields
        reserved_fields = {
            "model_type",
            "seed",
            "runtime",
            "pretrained",
            "pretrained_name",
            "params",
        }
        model_type = data.pop("model_type")
        seed = data.pop("seed", 42)
        runtime = data.pop("runtime", {})
        pretrained = data.pop("pretrained", False)
        pretrained_name = data.pop("pretrained_name", None)

        # All remaining fields go into params
        params = data

        return ModelConfig(
            model_type=model_type,
            seed=seed,
            params=params,
            runtime=runtime,
            pretrained=pretrained,
            pretrained_name=pretrained_name,
        )

    def to_yaml(self, path: str):
        """
        Save configuration to YAML file.

        Saves both reserved fields and all params to YAML.

        Args:
            path: Path to save YAML configuration file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build output dict with reserved fields first, then params
        output = {
            "model_type": self.model_type,
            "seed": self.seed,
        }
        # Add all params
        output.update(self.params)
        # Add runtime and pretrained info if present
        if self.runtime:
            output["runtime"] = self.runtime
        if self.pretrained:
            output["pretrained"] = self.pretrained
        if self.pretrained_name:
            output["pretrained_name"] = self.pretrained_name

        with open(path, "w") as f:
            yaml.safe_dump(output, f, default_flow_style=False, sort_keys=False)

    def save_to_pretrained(
        self,
        name: str,
        pretrained_dir: Optional[Union[str, Path]] = None,
        version: Optional[str] = None,
    ) -> Path:
        """
        Save configuration to the pretrained models directory.

        Supports both versioned and non-versioned saves:
        - Non-versioned: pretrained/<model_type>/<name>.yaml
        - Versioned: pretrained/<model_type>/<name>/<version>/<name>.yaml

        Args:
            name: Name to save the model as (e.g., "cifar10_ffn_baseline")
            pretrained_dir: Base directory for pretrained models.
                           If None, uses difflut/models/pretrained
            version: Optional version string (e.g., "v1").
                    If provided, saves to versioned location.

        Returns:
            Path to the saved configuration file

        Example:
            >>> config = ModelConfig(...)
            >>> # Non-versioned
            >>> config_path = config.save_to_pretrained("cifar10_ffn_baseline")
            >>> # Saves to: pretrained/feedforward/cifar10_ffn_baseline.yaml

            >>> # Versioned
            >>> config_path = config.save_to_pretrained("cifar10_ffn_baseline", version="v1")
            >>> # Saves to: pretrained/feedforward/cifar10_ffn_baseline/v1/cifar10_ffn_baseline.yaml
        """
        if pretrained_dir is None:
            # Use default pretrained directory in models package
            pretrained_dir = Path(__file__).parent / "pretrained"
        else:
            pretrained_dir = Path(pretrained_dir)

        # Create model-type-specific subdirectory
        model_type_dir = pretrained_dir / self.model_type
        model_type_dir.mkdir(parents=True, exist_ok=True)

        if version:
            # Versioned save: pretrained/<model_type>/<name>/<version>/
            version_dir = model_type_dir / name / version
            version_dir.mkdir(parents=True, exist_ok=True)
            config_path = version_dir / f"{name}.yaml"
        else:
            # Non-versioned save: pretrained/<model_type>/
            config_path = model_type_dir / f"{name}.yaml"

        self.to_yaml(str(config_path))

        return config_path

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of config (all params flattened)
        """
        result = {
            "model_type": self.model_type,
            "seed": self.seed,
        }
        result.update(self.params)
        if self.runtime:
            result["runtime"] = self.runtime
        if self.pretrained:
            result["pretrained"] = self.pretrained
        if self.pretrained_name:
            result["pretrained_name"] = self.pretrained_name
        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ModelConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary with configuration parameters

        Returns:
            ModelConfig instance
        """
        # Extract reserved fields
        model_type = data.get("model_type")
        seed = data.get("seed", 42)
        runtime = data.get("runtime", {})
        pretrained = data.get("pretrained", False)
        pretrained_name = data.get("pretrained_name", None)

        # Build params dict with remaining fields
        reserved = {"model_type", "seed", "runtime", "pretrained", "pretrained_name"}
        params = {k: v for k, v in data.items() if k not in reserved}

        return ModelConfig(
            model_type=model_type,
            seed=seed,
            params=params,
            runtime=runtime,
            pretrained=pretrained,
            pretrained_name=pretrained_name,
        )

    # ==================== Utility Methods ====================

    def get_param(self, name: str, default: Any = None) -> Any:
        """
        Get a parameter from the model config.

        Args:
            name: Parameter name
            default: Default value if not found

        Returns:
            Parameter value or default
        """
        return self.params.get(name, default)

    def get_runtime_param(self, name: str, default: Any = None) -> Any:
        """
        Get a runtime parameter.

        Args:
            name: Parameter name
            default: Default value if not found

        Returns:
            Parameter value or default
        """
        return self.runtime.get(name, default)

    def is_compatible_with_weights(self, other: "ModelConfig") -> bool:
        """
        Check if this config is compatible with weights from another config.

        Two configs are compatible if they have the same model_type and params.

        Args:
            other: Another ModelConfig instance

        Returns:
            True if configs are compatible (weights can be shared)
        """
        # Model type must match
        if self.model_type != other.model_type:
            return False

        # All params must match (weights are locked to architecture)
        if self.params != other.params:
            return False

        return True

    def __repr__(self) -> str:
        """String representation of config."""
        runtime_str = (
            ", ".join(f"{k}={v}" for k, v in self.runtime.items()) if self.runtime else "none"
        )
        params_str = ", ".join(f"{k}={v}" for k, v in list(self.params.items())[:3])  # Show first 3
        if len(self.params) > 3:
            params_str += ", ..."
        return (
            f"ModelConfig(\n"
            f"  model_type={self.model_type},\n"
            f"  params=[{params_str}],\n"
            f"  runtime=[{runtime_str}]\n"
            f")"
        )
