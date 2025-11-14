"""
Base model infrastructure for DiffLUT models.

Provides a template for all DiffLUT models with config management,
runtime parameter overrides, and standardized forward pass structure.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .model_config import ModelConfig


class BaseLUTModel(nn.Module):
    """
    Base class for all DiffLUT models.

    Features:
    - Config-based initialization for reproducibility
    - Runtime parameter overrides (safe to change without affecting structure)
    - Hooks for subclasses to respond to runtime changes
    - Standardized serialization and deserialization

    Subclasses should implement:
    - forward(): The actual forward pass logic
    - on_runtime_update(): React to runtime parameter changes (optional)
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the base model.

        Args:
            config: ModelConfig instance containing all model parameters
        """
        super().__init__()
        self.config = config
        self.runtime = config.runtime.copy() if config.runtime else {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor

        Raises:
            NotImplementedError: Subclasses must implement this
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def apply_runtime_overrides(self, overrides: Optional[Dict[str, Any]] = None):
        """
        Apply runtime-only parameter overrides.

        These are parameters that can be changed without affecting the model
        structure (e.g., dropout rate, bitflip probability, temperature).

        Args:
            overrides: Dictionary of runtime parameters to override
        """
        if overrides:
            self.runtime.update(overrides)
            self.on_runtime_update()

    def on_runtime_update(self):
        """
        Hook for subclasses to react to runtime parameter changes.

        This is called after runtime parameters are updated via
        apply_runtime_overrides(). Subclasses can override this to
        apply the changes to internal components.

        Example:
            def on_runtime_update(self):
                if "temperature" in self.runtime:
                    for layer in self.layers:
                        if hasattr(layer, 'temperature'):
                            layer.temperature = self.runtime["temperature"]
        """
        pass

    def get_config(self) -> ModelConfig:
        """
        Get the model configuration.

        Returns:
            ModelConfig instance with current configuration
        """
        return self.config

    def save_config(self, path: str):
        """
        Save model configuration to YAML file.

        Args:
            path: Path to save configuration file
        """
        self.config.to_yaml(path)

    def save_weights(self, path: str):
        """
        Save model weights to file.

        Args:
            path: Path to save weights file
        """
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, map_location: str = "cpu"):
        """
        Load model weights from file.

        Args:
            path: Path to weights file
            map_location: Device to map tensors to (default: "cpu")
        """
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)

    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.

        Returns:
            Dictionary with 'total', 'trainable', and 'non_trainable' counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {"total": total, "trainable": trainable, "non_trainable": total - trainable}

    def get_regularization_loss(self) -> torch.Tensor:
        """
        Compute total regularization loss from all layers.

        This is a default implementation that can be overridden by subclasses.

        Returns:
            Total regularization loss (scalar tensor)
        """
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        # Iterate through all submodules looking for regularization
        for module in self.modules():
            if hasattr(module, "get_regularization_loss") and module != self:
                reg_loss = reg_loss + module.get_regularization_loss()

        return reg_loss

    def __repr__(self) -> str:
        """String representation of the model."""
        param_count = self.count_parameters()
        return (
            f"{self.__class__.__name__}(\n"
            f"  config={self.config.__class__.__name__},\n"
            f"  parameters={param_count['total']:,} ({param_count['trainable']:,} trainable),\n"
            f"  runtime={self.runtime}\n"
            f")"
        )
