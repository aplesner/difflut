"""
Model factory for building DiffLUT models.

Provides a unified interface for creating models from:
- Pretrained model names
- YAML configuration files
- ModelConfig objects

Supports:
- Automatic weight loading
- Runtime parameter overrides
- Pretrained model discovery
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from ..registry import REGISTRY
from .base_model import BaseLUTModel
from .model_config import ModelConfig

# Default location for pretrained models
PRETRAINED_DIR = Path(__file__).parent / "pretrained"


def build_model(
    source: Union[str, ModelConfig],
    *,
    load_weights: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
    pretrained_dir: Optional[Union[str, Path]] = None,
) -> BaseLUTModel:
    """
    Build a DiffLUT model from various sources.

    This is the main entry point for creating models. It handles:
    - Pretrained models by name (with auto-versioning to latest)
    - YAML configuration files
    - ModelConfig objects
    - Optional weight loading
    - Runtime parameter overrides

    Args:
        source: One of:
            - Pretrained model name (e.g., "mnist_large" or "feedforward/mnist_large/v1")
              If version is omitted, loads the latest available version.
            - Path to YAML config file (e.g., "configs/my_model.yaml")
            - ModelConfig object
        load_weights: Whether to load pretrained weights (default: True)
                     Only applies to pretrained models
        overrides: Dictionary of runtime parameters to override
                  These must be runtime-safe (not structural) parameters
        pretrained_dir: Custom directory for pretrained models
                       If None, uses default location

    Returns:
        Initialized model (BaseLUTModel subclass)

    Examples:
        # Load pretrained model (latest version) with weights
        >>> model = build_model("mnist_large", load_weights=True)

        # Load specific version
        >>> model = build_model("feedforward/mnist_large/v1", load_weights=True)

        # Load architecture without weights (for training)
        >>> model = build_model("mnist_large", load_weights=False)

        # Override runtime parameters
        >>> model = build_model("mnist_large", overrides={"temperature": 0.5})

        # Build from YAML file
        >>> model = build_model("configs/my_model.yaml")

        # Build from config object
        >>> config = ModelConfig(...)
        >>> model = build_model(config)

    Raises:
        ValueError: If source is invalid or model type not found
        FileNotFoundError: If config or weight file not found
    """
    # Set pretrained directory
    if pretrained_dir is None:
        pretrained_dir = PRETRAINED_DIR
    else:
        pretrained_dir = Path(pretrained_dir)

    config = None
    weights_path = None

    # ==================== Case 1: Pretrained Model Name ====================
    if isinstance(source, str) and not os.path.isfile(source):
        # Try to find pretrained model
        config, weights_path = _load_pretrained_model(
            source, pretrained_dir, load_weights
        )

    # ==================== Case 2: YAML Configuration File ====================
    elif isinstance(source, str) and source.endswith((".yaml", ".yml")):
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Configuration file not found: {source}")

        config = ModelConfig.from_yaml(source)

        # Check if config specifies pretrained weights
        if config.pretrained and config.pretrained_name and load_weights:
            _, weights_path = _load_pretrained_model(
                config.pretrained_name, pretrained_dir, load_weights
            )

    # ==================== Case 3: ModelConfig Object ====================
    elif isinstance(source, ModelConfig):
        config = source

        # Check if config specifies pretrained weights
        if config.pretrained and config.pretrained_name and load_weights:
            _, weights_path = _load_pretrained_model(
                config.pretrained_name, pretrained_dir, load_weights
            )

    else:
        raise ValueError(
            f"Invalid source: {source}. "
            f"Expected pretrained model name, YAML path, or ModelConfig object."
        )

    # ==================== Build Model ====================

    # Apply runtime overrides to config before building
    if overrides:
        config.runtime.update(overrides)

    # Get model class from registry
    try:
        model_class = REGISTRY.get_model(config.model_type)
    except ValueError:
        raise ValueError(
            f"Model type '{config.model_type}' not found in registry. "
            f"Available types: {REGISTRY.list_models()}"
        )

    # Create model instance
    model = model_class(config)

    # Load weights if available
    if weights_path is not None and os.path.exists(weights_path):
        print(f"Loading pretrained weights from: {weights_path}")
        model.load_weights(weights_path)
    elif load_weights and config.pretrained:
        print(f"Warning: load_weights=True but no weights found for {source}")

    return model


def build_model_for_experiment(
    model_config: ModelConfig,
    *,
    load_from_pretrained: bool = False,
    save_as_pretrained: bool = False,
    pretrained_name: Optional[str] = None,
    runtime_overrides: Optional[Dict[str, Any]] = None,
    pretrained_dir: Optional[Union[str, Path]] = None,
) -> tuple[BaseLUTModel, Optional[Path]]:
    """
    Build a model for experiment with pretrained handling.

    This is the main entry point for building models in experiments.
    It handles the pretrained model workflow:
    - If load_from_pretrained: Load config and weights from pretrained, apply runtime_overrides
    - If save_as_pretrained: Return path where weights should be saved

    Args:
        model_config: ModelConfig object from experiment config
        load_from_pretrained: Whether to load from pretrained models
        save_as_pretrained: Whether to save this model to pretrained after training
        pretrained_name: Name of pretrained model (e.g., "feedforward/cifar10_ffn_baseline")
        runtime_overrides: Runtime parameters to override (from experiment config)
        pretrained_dir: Custom directory for pretrained models

    Returns:
        Tuple of (model, weights_save_path)
        - model: Initialized model instance
        - weights_save_path: Path where weights should be saved (None if save_as_pretrained=False)

    Raises:
        ValueError: If pretrained model not found or config invalid
    """
    if pretrained_dir is None:
        pretrained_dir = PRETRAINED_DIR
    else:
        pretrained_dir = Path(pretrained_dir)

    weights_save_path = None

    # ==================== Case 1: Load from Pretrained ====================
    if load_from_pretrained and pretrained_name:
        config, weights_path = _load_pretrained_model(
            pretrained_name, pretrained_dir, load_weights=True
        )

        # Override runtime parameters from experiment config
        if runtime_overrides:
            config.runtime.update(runtime_overrides)

        print(f"Loaded pretrained model: {pretrained_name}")
        if weights_path and os.path.exists(weights_path):
            print(f"  Config: {pretrained_name}.yaml")
            print(f"  Weights: {weights_path}")

    # ==================== Case 2: Use Experiment Config ====================
    else:
        config = model_config

        # Apply runtime overrides if provided
        if runtime_overrides:
            config.runtime.update(runtime_overrides)

    # ==================== Build Model ====================

    # Get model class from registry
    try:
        model_class = REGISTRY.get_model(config.model_type)
    except ValueError:
        raise ValueError(
            f"Model type '{config.model_type}' not found in registry. "
            f"Available types: {REGISTRY.list_models()}"
        )

    # Create model instance
    model = model_class(config)

    # ==================== Load Weights if Pretrained ====================

    if load_from_pretrained and pretrained_name:
        _, weights_path = _load_pretrained_model(
            pretrained_name, pretrained_dir, load_weights=True
        )
        if weights_path and os.path.exists(weights_path):
            print(f"Loading pretrained weights from: {weights_path}")
            model.load_weights(weights_path)

    # ==================== Setup Save Path if Needed ====================

    if save_as_pretrained and pretrained_name:
        # Parse pretrained_name to extract model_type, model_name, and version
        # Format: model_type/model_name[/version]
        pretrained_parts = pretrained_name.split("/")

        if len(pretrained_parts) == 2:
            model_type, model_name = pretrained_parts
            version = None
        elif len(pretrained_parts) == 3:
            model_type, model_name, version = pretrained_parts
        else:
            raise ValueError(
                f"pretrained_name must be in format 'model_type/model_name' or "
                f"'model_type/model_name/version', got: {pretrained_name}"
            )

        # Prepare save path and config
        weights_save_path, config_save_path = _prepare_pretrained_save_paths(
            pretrained_dir, model_type, model_name, version, config
        )

        print(f"Will save model to pretrained:")
        print(f"  Config: {config_save_path}")
        print(f"  Weights: {weights_save_path}")

    return model, weights_save_path


def _load_pretrained_model(
    name: str, pretrained_dir: Path, load_weights: bool
) -> tuple[ModelConfig, Optional[Path]]:
    """
    Load a pretrained model by name (with automatic latest version selection).

    Searches for model in versioned directory structure:
    - pretrained/<model_type>/<name>/<version>/<name>.yaml

    Supports multiple name formats:
    - Simple name: "mnist_large" (loads latest version across all model types)
    - With model type: "feedforward/mnist_large" (loads latest version of that model)
    - With version: "feedforward/mnist_large/v1" (loads specific version)

    Args:
        name: Name of pretrained model (optionally with model_type and/or version)
        pretrained_dir: Directory containing pretrained models
        load_weights: Whether to look for weight file

    Returns:
        Tuple of (config, weights_path)
        weights_path is None if not found or load_weights=False

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If name format is invalid
    """
    # Parse name to extract model_type, model_name, and version
    # Format: [model_type/]model_name[/version]
    parts = name.split("/")
    model_type = None
    model_name = None
    version = None

    if len(parts) == 1:
        # Format: model_name (no model_type or version)
        model_name = parts[0]
    elif len(parts) == 2:
        # Format: model_type/model_name OR model_name/version
        # Try model_type/model_name first
        potential_type_dir = pretrained_dir / parts[0]
        if potential_type_dir.is_dir():
            model_type = parts[0]
            model_name = parts[1]
        else:
            # Otherwise assume model_name/version
            model_name = parts[0]
            version = parts[1]
    elif len(parts) == 3:
        # Format: model_type/model_name/version
        model_type = parts[0]
        model_name = parts[1]
        version = parts[2]
    else:
        raise ValueError(
            f"Invalid pretrained model name format: {name}. "
            f"Expected: [model_type/]model_name[/version]"
        )

    config_path = None
    weights_path = None

    # If model_type is specified, search only in that directory
    if model_type:
        model_type_dir = pretrained_dir / model_type
        if not model_type_dir.is_dir():
            raise FileNotFoundError(f"Model type directory not found: {model_type_dir}")
        config_path, weights_path = _find_model_in_dir(
            model_type_dir, model_name, version, load_weights
        )
    else:
        # Search across all model_type directories
        for model_type_dir in pretrained_dir.iterdir():
            if not model_type_dir.is_dir():
                continue

            try:
                config_path, weights_path = _find_model_in_dir(
                    model_type_dir, model_name, version, load_weights
                )
                break
            except FileNotFoundError:
                continue

    if config_path is None:
        raise FileNotFoundError(
            f"Pretrained model '{name}' not found in {pretrained_dir}. "
            f"Available models: {list_pretrained_models(pretrained_dir)}"
        )

    # Load config
    config = ModelConfig.from_yaml(str(config_path))

    # Ensure pretrained info is set
    config.pretrained = True
    if config.pretrained_name is None:
        config.pretrained_name = name

    return config, weights_path


def _find_model_in_dir(
    model_type_dir: Path, model_name: str, version: Optional[str], load_weights: bool
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Find versioned model config and weights in a model type directory.

    Searches versioned locations only:
    - model_type_dir / model_name / version / model_name.yaml (if version specified)
    - model_type_dir / model_name / (latest) / model_name.yaml (if version not specified)

    Args:
        model_type_dir: Directory for a specific model type
        model_name: Name of the model
        version: Optional version string (e.g., "v1"). If None, uses latest version
        load_weights: Whether to look for weight files

    Returns:
        Tuple of (config_path, weights_path)

    Raises:
        FileNotFoundError: If model not found in any version
    """
    model_name_dir = model_type_dir / model_name

    if not model_name_dir.is_dir():
        raise FileNotFoundError(f"Model '{model_name}' not found in {model_type_dir}")

    if version:
        # Search for specific version
        version_dir = model_name_dir / version
    else:
        # Find latest version
        versions = sorted(
            [d.name for d in model_name_dir.iterdir() if d.is_dir()],
            key=lambda v: _version_sort_key(v),
        )
        if not versions:
            raise FileNotFoundError(
                f"No versions found for model '{model_name}' in {model_type_dir}"
            )
        latest_version = versions[-1]
        version_dir = model_name_dir / latest_version

    config_path = version_dir / f"{model_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found for model '{model_name}' in {version_dir}"
        )

    weights_path = None
    if load_weights:
        weight_candidate = version_dir / f"{model_name}.pth"
        if weight_candidate.exists():
            weights_path = weight_candidate

    return config_path, weights_path


def _version_sort_key(version: str) -> tuple:
    """
    Generate a sort key for version strings like 'v1', 'v2', etc.

    Handles both numeric versions (v1, v2, ...) and arbitrary strings.

    Args:
        version: Version string

    Returns:
        Tuple for sorting (higher versions sort later)
    """
    # Extract numeric part if it starts with 'v'
    if version.startswith("v") and version[1:].isdigit():
        return (0, int(version[1:]))
    else:
        # Non-standard version strings sort before numeric ones
        return (1, version)


def _prepare_pretrained_save_paths(
    pretrained_dir: Path,
    model_type: str,
    model_name: str,
    version: Optional[str],
    config: ModelConfig,
) -> tuple[Path, Path]:
    """
    Prepare paths for saving pretrained model and config (always versioned).

    All models are saved in versioned directories:
    - pretrained/<model_type>/<model_name>/<version>/<model_name>.yaml and .pth

    If no version is specified, automatically finds the next available version (v1, v2, ...).

    Args:
        pretrained_dir: Base pretrained models directory
        model_type: Model type (e.g., "feedforward")
        model_name: Model name (e.g., "cifar10_ffn_baseline")
        version: Optional version string (e.g., "v1"). If None, auto-increments to next version
        config: ModelConfig object to save

    Returns:
        Tuple of (weights_save_path, config_save_path)
    """
    model_type_dir = pretrained_dir / model_type
    model_type_dir.mkdir(parents=True, exist_ok=True)

    model_name_dir = model_type_dir / model_name

    if version is None:
        # Auto-increment: find next available version
        if model_name_dir.exists() and model_name_dir.is_dir():
            # Find all version directories
            versions = sorted(
                [d.name for d in model_name_dir.iterdir() if d.is_dir()],
                key=lambda v: _version_sort_key(v),
            )
            if versions:
                latest_version = versions[-1]
                # Increment version number
                if latest_version.startswith("v") and latest_version[1:].isdigit():
                    next_version_num = int(latest_version[1:]) + 1
                    version = f"v{next_version_num}"
                else:
                    # Fallback: append version counter
                    version = f"{latest_version}_v1"
            else:
                version = "v1"
        else:
            # First time: start with v1
            version = "v1"

    # Create versioned directory
    version_dir = model_name_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    weights_save_path = version_dir / f"{model_name}.pth"
    config_save_path = version_dir / f"{model_name}.yaml"

    # Save config to the prepared path
    config.to_yaml(str(config_save_path))

    return weights_save_path, config_save_path


def list_pretrained_models(
    pretrained_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, list[str]]:
    """
    List all available pretrained models.

    Returns models in both formats (non-versioned and versioned):
    - Non-versioned: "model_name"
    - Versioned: "model_name/v1", "model_name/v2", etc.

    Args:
        pretrained_dir: Directory containing pretrained models
                       If None, uses default location

    Returns:
        Dictionary mapping model type to list of available model names
        Example: {
            "feedforward": [
                "mnist_large",                          # non-versioned
                "cifar10_ffn_baseline/v1",              # versioned
                "cifar10_ffn_baseline/v2",
            ]
        }
    """
    if pretrained_dir is None:
        pretrained_dir = PRETRAINED_DIR
    else:
        pretrained_dir = Path(pretrained_dir)

    if not pretrained_dir.exists():
        return {}

    models = {}

    for model_type_dir in pretrained_dir.iterdir():
        if not model_type_dir.is_dir():
            continue

        model_type = model_type_dir.name
        model_names = []

        # Look for non-versioned models (*.yaml in model_type_dir)
        for config_file in model_type_dir.glob("*.yaml"):
            # Ignore README files
            if config_file.name.upper() == "README.MD":
                continue

            model_name = config_file.stem
            model_names.append(model_name)

        # Look for versioned models (model_type_dir/model_name/version/model_name.yaml)
        for model_name_dir in model_type_dir.iterdir():
            if not model_name_dir.is_dir():
                continue

            model_name = model_name_dir.name

            # Skip if this is a versioned structure (skip if parent dir name is the config file stem)
            for version_dir in model_name_dir.iterdir():
                if not version_dir.is_dir():
                    continue

                version = version_dir.name
                config_file = version_dir / f"{model_name}.yaml"
                if config_file.exists():
                    # Add versioned name
                    versioned_name = f"{model_name}/{version}"
                    model_names.append(versioned_name)

        if model_names:
            models[model_type] = sorted(model_names)

    return models


def get_pretrained_model_info(
    name: str, pretrained_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Get information about a pretrained model.

    Supports both versioned and non-versioned names:
    - "cifar10_ffn_baseline"
    - "feedforward/cifar10_ffn_baseline/v1"

    Args:
        name: Name of pretrained model
        pretrained_dir: Directory containing pretrained models

    Returns:
        Dictionary with model information:
        - config: ModelConfig object
        - has_weights: Whether weights file exists
        - config_path: Path to config file
        - weights_path: Path to weights file (if exists)
        - model_type: Model type (e.g., "feedforward")
        - layer_widths: Layer widths from config
        - num_classes: Number of output classes
        - dataset: Dataset name if available

    Raises:
        FileNotFoundError: If model not found
    """
    if pretrained_dir is None:
        pretrained_dir = PRETRAINED_DIR
    else:
        pretrained_dir = Path(pretrained_dir)

    config, weights_path = _load_pretrained_model(
        name, pretrained_dir, load_weights=True
    )

    # Find config path by trying all possible locations
    config_path = None

    # Parse name for version info
    parts = name.split("/")
    if len(parts) == 1:
        model_name = parts[0]
        # Search across all model types for non-versioned
        for model_type_dir in pretrained_dir.iterdir():
            if not model_type_dir.is_dir():
                continue
            candidate = model_type_dir / f"{model_name}.yaml"
            if candidate.exists():
                config_path = candidate
                break
    elif len(parts) == 2:
        potential_model_type = parts[0]
        model_name = parts[1]
        # Try as model_type/model_name
        potential_dir = pretrained_dir / potential_model_type
        if potential_dir.is_dir():
            candidate = potential_dir / f"{model_name}.yaml"
            if candidate.exists():
                config_path = candidate
        if not config_path:
            # Try as model_name/version
            for model_type_dir in pretrained_dir.iterdir():
                if not model_type_dir.is_dir():
                    continue
                candidate = (
                    model_type_dir
                    / model_name
                    / potential_model_type
                    / f"{model_name}.yaml"
                )
                if candidate.exists():
                    config_path = candidate
                    break
    elif len(parts) == 3:
        model_type = parts[0]
        model_name = parts[1]
        version = parts[2]
        # model_type/model_name/version
        model_type_dir = pretrained_dir / model_type
        if model_type_dir.is_dir():
            candidate = model_type_dir / model_name / version / f"{model_name}.yaml"
            if candidate.exists():
                config_path = candidate

    return {
        "config": config,
        "has_weights": weights_path is not None and weights_path.exists(),
        "config_path": config_path,
        "weights_path": weights_path,
        "model_type": config.model_type,
        "layer_widths": config.layer_widths,
        "num_classes": config.num_classes,
        "dataset": config.dataset,
    }


# Convenience function for backward compatibility
def load_pretrained(name: str, **kwargs) -> BaseLUTModel:
    """
    Load a pretrained model (convenience function).

    This is equivalent to build_model(name, load_weights=True, **kwargs)

    Args:
        name: Name of pretrained model
        **kwargs: Additional arguments passed to build_model

    Returns:
        Initialized model with pretrained weights
    """
    return build_model(name, load_weights=True, **kwargs)
