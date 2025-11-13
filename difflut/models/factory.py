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
import torch
from pathlib import Path
from typing import Union, Optional, Dict, Any

from .model_config import ModelConfig
from .base_model import BaseLUTModel
from ..registry import REGISTRY


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
    - Pretrained models by name
    - YAML configuration files
    - ModelConfig objects
    - Optional weight loading
    - Runtime parameter overrides
    
    Args:
        source: One of:
            - Pretrained model name (e.g., "mnist_large")
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
        # Load pretrained model with weights
        >>> model = build_model("mnist_large", load_weights=True)
        
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


def _load_pretrained_model(
    name: str,
    pretrained_dir: Path,
    load_weights: bool
) -> tuple[ModelConfig, Optional[Path]]:
    """
    Load a pretrained model by name.
    
    Searches for model in pretrained directory structure:
    - pretrained/<model_type>/<name>.yaml
    - pretrained/<model_type>/<name>.pth
    
    Args:
        name: Name of pretrained model
        pretrained_dir: Directory containing pretrained models
        load_weights: Whether to look for weight file
    
    Returns:
        Tuple of (config, weights_path)
        weights_path is None if not found or load_weights=False
    
    Raises:
        FileNotFoundError: If config file not found
    """
    # Try to find config file
    # Search pattern: pretrained/*/<name>.yaml
    config_path = None
    weights_path = None
    
    for model_type_dir in pretrained_dir.iterdir():
        if not model_type_dir.is_dir():
            continue
        
        candidate = model_type_dir / f"{name}.yaml"
        if candidate.exists():
            config_path = candidate
            
            # Check for weights in same directory
            if load_weights:
                weight_candidate = model_type_dir / f"{name}.pth"
                if weight_candidate.exists():
                    weights_path = weight_candidate
            
            break
    
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


def list_pretrained_models(pretrained_dir: Optional[Union[str, Path]] = None) -> Dict[str, list[str]]:
    """
    List all available pretrained models.
    
    Args:
        pretrained_dir: Directory containing pretrained models
                       If None, uses default location
    
    Returns:
        Dictionary mapping model type to list of available model names
        Example: {"feedforward": ["mnist_large", "mnist_small"], "convnet": [...]}
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
        
        for config_file in model_type_dir.glob("*.yaml"):
            # Ignore README files
            if config_file.name.upper() == "README.MD":
                continue
            
            model_name = config_file.stem
            model_names.append(model_name)
        
        if model_names:
            models[model_type] = sorted(model_names)
    
    return models


def get_pretrained_model_info(
    name: str,
    pretrained_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Get information about a pretrained model.
    
    Args:
        name: Name of pretrained model
        pretrained_dir: Directory containing pretrained models
    
    Returns:
        Dictionary with model information:
        - config: ModelConfig object
        - has_weights: Whether weights file exists
        - config_path: Path to config file
        - weights_path: Path to weights file (if exists)
    
    Raises:
        FileNotFoundError: If model not found
    """
    if pretrained_dir is None:
        pretrained_dir = PRETRAINED_DIR
    else:
        pretrained_dir = Path(pretrained_dir)
    
    config, weights_path = _load_pretrained_model(name, pretrained_dir, load_weights=True)
    
    # Find config path
    config_path = None
    for model_type_dir in pretrained_dir.iterdir():
        if not model_type_dir.is_dir():
            continue
        candidate = model_type_dir / f"{name}.yaml"
        if candidate.exists():
            config_path = candidate
            break
    
    return {
        'config': config,
        'has_weights': weights_path is not None and weights_path.exists(),
        'config_path': config_path,
        'weights_path': weights_path,
        'model_type': config.model_type,
        'layer_widths': config.layer_widths,
        'num_classes': config.num_classes,
        'dataset': config.dataset,
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
