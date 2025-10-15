import torch
from .base_encoder import BaseEncoder
from ..registry import register_encoder

@register_encoder("thermometer")
class ThermometerEncoder(BaseEncoder):
    """
    Thermometer encoding: represents a value using multiple thresholds.
    Each threshold produces a binary bit, creating a thermometer-like pattern.
    
    Example: value=0.6 with 3 bits and thresholds [0.25, 0.5, 0.75] -> [1, 1, 0]
    """
    
    def __init__(self, num_bits: int = 1, feature_wise: bool = True):
        """
        Args:
            num_bits: Number of threshold bits
            feature_wise: If True, compute thresholds per feature; if False, global thresholds
        """
        super().__init__(num_bits=num_bits)
        assert isinstance(feature_wise, bool), "feature_wise must be a boolean"
        self.feature_wise = feature_wise
        self.thresholds = None
    
    def _compute_thresholds(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute evenly spaced thresholds between min and max values.
        
        Args:
            x: Input tensor
            
        Returns:
            Threshold tensor
        """
        # Compute min/max per feature or globally
        if self.feature_wise:
            min_value = x.min(dim=0)[0]
            max_value = x.max(dim=0)[0]
        else:
            min_value = x.min()
            max_value = x.max()
        
        # Create evenly spaced thresholds
        # Shape: (num_features, num_bits) if feature_wise, else (1, num_bits)
        threshold_indices = torch.arange(1, self.num_bits + 1, device=x.device)
        step_size = (max_value - min_value) / (self.num_bits + 1)
        
        if self.feature_wise:
            thresholds = min_value.unsqueeze(-1) + threshold_indices.unsqueeze(0) * step_size.unsqueeze(-1)
        else:
            thresholds = min_value + threshold_indices * step_size
        
        return thresholds
    
    def fit(self, x: torch.Tensor) -> 'ThermometerEncoder':
        """
        Fit the encoder by computing thresholds from the data.
        
        Args:
            x: Input data tensor
            
        Returns:
            self for method chaining
        """
        x = self._to_tensor(x)
        self.thresholds = self._compute_thresholds(x)
        self._is_fitted = True
        return self
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using thermometer encoding.
        
        Args:
            x: Input tensor to encode
            
        Returns:
            Binary encoded tensor with shape (batch_size, num_features * num_bits)
            For feature-wise encoding, the output is flattened so each feature's bits are consecutive.
        """
        self._check_fitted()
        x = self._to_tensor(x)
        
        # Add dimension for broadcasting with thresholds
        x_expanded = x.unsqueeze(-1)
        
        # Compare with thresholds to get binary encoding
        # Shape: (batch_size, num_features, num_bits) for feature-wise
        #    or: (batch_size, num_bits) for global
        encoded = (x_expanded > self.thresholds).float()
        
        # Flatten the last two dimensions for feature-wise encoding
        # This converts (batch_size, num_features, num_bits) -> (batch_size, num_features * num_bits)
        if self.feature_wise and encoded.dim() == 3:
            batch_size = encoded.shape[0]
            encoded = encoded.reshape(batch_size, -1)
        
        return encoded
    
    # Backward compatibility aliases
    def binarize(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for encode() for backward compatibility."""
        return self.encode(x)
    
    def get_thresholds(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for _compute_thresholds() for backward compatibility."""
        return self._compute_thresholds(x)
    
    def __repr__(self) -> str:
        return f"ThermometerEncoder(num_bits={self.num_bits}, feature_wise={self.feature_wise})"


@register_encoder("gaussian_thermometer")
class GaussianThermometerEncoder(ThermometerEncoder):
    """
    Gaussian Thermometer encoding: uses Gaussian distribution quantiles for thresholds.
    Thresholds are placed at inverse CDF values assuming normal distribution.
    
    Better for data that follows a Gaussian distribution.
    """
    
    def _compute_thresholds(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute thresholds based on Gaussian quantiles.
        
        Args:
            x: Input tensor
            
        Returns:
            Threshold tensor
        """
        # Compute quantiles for Gaussian distribution
        quantile_positions = torch.arange(1, self.num_bits + 1, device=x.device).float() / (self.num_bits + 1)
        std_skews = torch.distributions.Normal(0, 1).icdf(quantile_positions)
        
        # Compute mean and std per feature or globally
        if self.feature_wise:
            mean = x.mean(dim=0)
            std = x.std(dim=0)
        else:
            mean = x.mean()
            std = x.std()
        
        # Compute thresholds: threshold = mean + std_skew * std
        # Stack thresholds along last dimension
        if self.feature_wise:
            thresholds = torch.stack([std_skew * std + mean for std_skew in std_skews], dim=-1)
        else:
            thresholds = std_skews * std + mean
        
        return thresholds
    
    def __repr__(self) -> str:
        return f"GaussianThermometerEncoder(num_bits={self.num_bits}, feature_wise={self.feature_wise})"


@register_encoder("distributive_thermometer")
class DistributiveThermometerEncoder(ThermometerEncoder):
    """
    Distributive Thermometer encoding: uses data distribution quantiles for thresholds.
    Thresholds are placed at actual data quantiles, ensuring balanced representation.
    
    Better for arbitrary data distributions as it adapts to the actual data.
    """
    
    def _compute_thresholds(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute thresholds based on data quantiles.
        
        Args:
            x: Input tensor
            
        Returns:
            Threshold tensor
        """
        if self.feature_wise:
            # Sort along the sample dimension (dim=0)
            data_sorted = torch.sort(x, dim=0)[0]
            num_samples = data_sorted.shape[0]
            
            # Compute indices for quantiles
            indices = torch.tensor(
                [int(num_samples * i / (self.num_bits + 1)) for i in range(1, self.num_bits + 1)],
                device=x.device
            )
            
            # Get thresholds at quantile positions
            thresholds = data_sorted[indices]  # Shape: (num_bits, num_features)
            
            # Permute to (num_features, num_bits)
            thresholds = thresholds.permute(*list(range(1, thresholds.ndim)), 0)
        else:
            # Flatten and sort all data
            data_sorted = torch.sort(x.flatten())[0]
            num_samples = data_sorted.shape[0]
            
            # Compute indices for quantiles
            indices = torch.tensor(
                [int(num_samples * i / (self.num_bits + 1)) for i in range(1, self.num_bits + 1)],
                device=x.device
            )
            
            # Get thresholds at quantile positions
            thresholds = data_sorted[indices]  # Shape: (num_bits,)
        
        return thresholds
    
    def __repr__(self) -> str:
        return f"DistributiveThermometerEncoder(num_bits={self.num_bits}, feature_wise={self.feature_wise})"


