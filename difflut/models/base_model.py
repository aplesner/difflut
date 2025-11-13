"""
Base class for DiffLUT model zoo models.

Provides common functionality for all models:
- Encoder fitting
- Regularization loss computation
- Parameter counting
- Model serialization
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class BaseModel(nn.Module, ABC):
    """
    Base class for all DiffLUT model zoo models.
    
    This class provides:
    - Common encoder fitting workflow
    - Regularization loss aggregation from all layers
    - Parameter counting utilities
    - Model metadata tracking
    
    Subclasses must implement:
    - fit_encoder(): Prepare encoder on training data
    - forward(): Model inference
    - _build_layers(): Create network architecture
    """
    
    def __init__(self, name: str, input_size: int, num_classes: int):
        """
        Initialize base model.
        
        Args:
            name: Model name for identification (e.g., 'mnist_fc_8k_linear')
            input_size: Size of raw input features
            num_classes: Number of output classes
        """
        super().__init__()
        self.model_name = name
        self.input_size = input_size
        self.num_classes = num_classes
        self.encoder_fitted = False
    
    @abstractmethod
    def fit_encoder(self, data: torch.Tensor) -> None:
        """
        Fit the encoder on training data.
        
        This is a required step before training the model, as it determines
        the number of encoded features that become the network input.
        
        Args:
            data: Training data tensor (N, ...) where ... represents feature dimensions
        
        Raises:
            RuntimeError: If encoder is already fitted
        """
        pass
    
    @abstractmethod
    def _build_layers(self) -> None:
        """
        Build the network layers after encoder is fitted.
        
        This is called automatically by fit_encoder() and should create:
        - self.layers: nn.ModuleList of DiffLUT layers
        - self.output_layer: Final classification layer (GroupSum)
        - self.layer_sizes: List tracking layer dimensions
        """
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (N, ...) - raw input (images, sequences, etc.)
        
        Returns:
            Output logits (N, num_classes)
        
        Raises:
            RuntimeError: If encoder is not fitted
        """
        if not self.encoder_fitted:
            raise RuntimeError(
                f"Encoder must be fitted before forward pass. "
                f"Call {self.__class__.__name__}.fit_encoder(data) first."
            )
        return self._forward_impl(x)
    
    @abstractmethod
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        Actual forward pass implementation (called by forward()).
        
        Subclasses implement this instead of forward().
        
        Args:
            x: Input tensor (N, ...) - raw input
        
        Returns:
            Output logits (N, num_classes)
        """
        pass
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Compute total regularization loss from all layers.
        
        Aggregates regularization contributions from all DiffLUT nodes
        across all layers in the model.
        
        Returns:
            Total regularization loss (scalar tensor, or 0 if no regularizers)
        """
        if not hasattr(self, 'layers'):
            return torch.tensor(0.0, device=self.get_device())
        
        reg_loss = 0.0
        for layer in self.layers:
            if hasattr(layer, 'get_regularization_loss'):
                reg_loss = reg_loss + layer.get_regularization_loss()
        
        # Ensure it's a tensor
        if isinstance(reg_loss, (int, float)):
            return torch.tensor(reg_loss, device=self.get_device())
        return reg_loss
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.
        
        Returns:
            Dictionary with 'total', 'trainable', and 'non_trainable' counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable
        }
    
    def get_layer_topology(self) -> List[Dict[str, Any]]:
        """
        Get the layer topology for debugging/visualization.
        
        Returns:
            List of dicts with 'input' and 'output' sizes for each layer
        """
        if hasattr(self, 'layer_sizes'):
            return self.layer_sizes
        return []
    
    def get_device(self) -> torch.device:
        """Get the device this model is on."""
        return next(self.parameters()).device
    
    def save_checkpoint(self, path: Path) -> None:
        """
        Save model checkpoint with config.
        
        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state': self.state_dict(),
            'model_name': self.model_name,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'encoder_fitted': self.encoder_fitted,
        }
        
        if hasattr(self, 'encoder'):
            checkpoint['encoder_state'] = self.encoder.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.get_device())
        self.load_state_dict(checkpoint['model_state'])
        self.encoder_fitted = checkpoint.get('encoder_fitted', True)
        
        if 'encoder_state' in checkpoint and hasattr(self, 'encoder'):
            self.encoder.load_state_dict(checkpoint['encoder_state'])
    
    def __repr__(self) -> str:
        """String representation of model."""
        param_counts = self.count_parameters()
        return (
            f"{self.model_name}(\n"
            f"  input_size={self.input_size},\n"
            f"  num_classes={self.num_classes},\n"
            f"  encoder_fitted={self.encoder_fitted},\n"
            f"  parameters={param_counts['total']:,}\n"
            f")"
        )
