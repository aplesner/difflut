import torch
from abc import ABC, abstractmethod
from typing import Union, Optional
import warnings

class BaseEncoder(ABC):
    """
    Abstract base class for all encoders.
    Encoders transform continuous values into binary representations.
    
    Supports both flattened and unflattened output formats:
    - flatten=True (default): Returns 2D tensor (batch_size, input_dim * num_bits)
    - flatten=False: Returns 3D tensor (batch_size, input_dim, num_bits)
    """
    
    def __init__(self, num_bits: int = 1, flatten: bool = True):
        """
        Args:
            num_bits: Number of bits in the encoded representation
            flatten: If True, flatten output to 2D (batch_size, input_dim * num_bits).
                     If False, keep as 3D (batch_size, input_dim, num_bits).
                     Default: True
        """
        assert num_bits > 0, "num_bits must be positive"
        assert isinstance(flatten, bool), "flatten must be a boolean"
        self.num_bits = int(num_bits)
        self.flatten = flatten
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, x: Union[torch.Tensor, any]) -> 'BaseEncoder':
        """
        Fit the encoder to data (e.g., compute thresholds, statistics).
        
        Args:
            x: Input data to fit on
            
        Returns:
            self for method chaining
        """
        pass
    
    @abstractmethod
    def encode(self, x: Union[torch.Tensor, any]) -> torch.Tensor:
        """
        Encode continuous input to binary representation.
        Must call fit() before encode().
        
        Args:
            x: Input data to encode
            
        Returns:
            Binary encoded tensor
        """
        pass
    
    def _to_tensor(self, x: Union[torch.Tensor, any]) -> torch.Tensor:
        """
        Helper to convert input to torch.Tensor if needed.
        
        Args:
            x: Input data
            
        Returns:
            Torch tensor
        """
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x)
        return x
    
    def _check_fitted(self):
        """Check if encoder has been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before encoding. "
                "Call fit() first with your training data to compute encoding parameters. "
                "Example: encoder.fit(train_data).encode(test_data)"
            )
    
    @property
    def is_fitted(self) -> bool:
        """Check if encoder has been fitted."""
        return self._is_fitted
    
    def to(self, device: Union[str, torch.device]) -> 'BaseEncoder':
        """
        Move all tensor attributes to the specified device.
        
        Args:
            device: Target device
            
        Returns:
            self for method chaining
        """
        # Move all tensor attributes to device
        for attr_name in dir(self):
            if not attr_name.startswith('_'):
                attr = getattr(self, attr_name)
                if isinstance(attr, torch.Tensor):
                    setattr(self, attr_name, attr.to(device))
        return self
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_bits={self.num_bits}, flatten={self.flatten})"
