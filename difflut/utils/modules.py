import torch
import torch.nn as nn
import warnings
from typing import Optional
from .warnings import warn_default_value

# Default number of output groups (typically number of classes)
# GroupSum will divide input features into k groups and sum within each group
DEFAULT_GROUPSUM_K: int = 1
# Default temperature/tau parameter for scaling grouped output
# Divides the grouped sum by tau to adjust output magnitude
DEFAULT_GROUPSUM_TAU: float = 1.0
# Default flag for whether to randomly permute input features before grouping
# If True, adds randomness to which features are grouped together
DEFAULT_GROUPSUM_USE_RANDPERM: bool = False


class GroupSum(nn.Module):
    """
    Groups input features and sums them, dividing by tau.
    Fixed reshaping logic for proper grouping.
    
    Expected input shape: (batch_size, num_nodes * num_output_per_node)
    where num_nodes is the number of nodes from the previous layer and
    num_output_per_node is the number of outputs per node (typically 1).
    
    Output shape: (batch_size, k)
    where k is the number of groups (typically num_classes).
    
    The input is reshaped to (batch_size, k, group_size) where
    group_size = num_features // k, then summed across groups.
    """
    def __init__(
        self,
        k: Optional[int] = None,
        tau: Optional[float] = None,
        use_randperm: Optional[bool] = None,
    ) -> None:
        """
        Args:
            k: Number of output groups (number of classes). If None, uses DEFAULT_GROUPSUM_K.
            tau: Temperature parameter for scaling output. If None, uses DEFAULT_GROUPSUM_TAU.
            use_randperm: If True, randomly permute input features before grouping. 
                         If None, uses DEFAULT_GROUPSUM_USE_RANDPERM.
        """
        super().__init__()
        
        # Set k with default and warning
        if k is None:
            self.k = DEFAULT_GROUPSUM_K
            warn_default_value("k (GroupSum)", self.k, stacklevel=2)
        else:
            self.k = k
        
        # Set tau with default and warning
        if tau is None:
            self.tau = DEFAULT_GROUPSUM_TAU
            warn_default_value("tau (GroupSum)", self.tau, stacklevel=2)
        else:
            self.tau = tau
        
        # Set use_randperm with default and warning
        if use_randperm is None:
            self.use_randperm = DEFAULT_GROUPSUM_USE_RANDPERM
            warn_default_value("use_randperm (GroupSum)", self.use_randperm, stacklevel=2)
        else:
            self.use_randperm = use_randperm
        
        # Validate parameters
        if not isinstance(self.k, int) or self.k <= 0:
            raise ValueError(f"k must be a positive integer, got {self.k}")
        if not isinstance(self.tau, (int, float)) or self.tau <= 0:
            raise ValueError(f"tau must be a positive number, got {self.tau}")
        
    def _validate_input_dim(self, x: torch.Tensor) -> None:
        """
        Validate that input dimensions are compatible.
        
        Args:
            x: Input tensor
            
        Raises:
            ValueError: If input shape is invalid
        """
        if x.dim() != 2:
            raise ValueError(
                f"GroupSum expects 2D input (batch_size, num_features), "
                f"but got shape {x.shape} with {x.dim()} dimensions. "
                f"Ensure the input comes from a Layer which should output "
                f"(batch_size, num_nodes * num_output_per_node)."
            )
        
        batch_size, num_features = x.shape
        
        if batch_size == 0:
            raise ValueError(
                f"GroupSum requires non-empty batch, got batch_size={batch_size}"
            )
        
        if num_features == 0:
            raise ValueError(
                f"GroupSum requires non-zero features, got num_features={num_features}. "
                f"This suggests the input layer has no outputs."
            )
        
        if num_features < self.k:
            warnings.warn(
                f"GroupSum input has fewer features ({num_features}) than output groups ({self.k}). "
                f"Some groups will receive zero features. This may be intentional for "
                f"hierarchical grouping, but check that your layer configuration is correct.",
                UserWarning,
                stacklevel=3
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: group input features and sum within groups.
        
        Args:
            x: Input tensor of shape (batch_size, num_features)
               Should be output from a Layer: (batch_size, num_nodes * num_output_per_node)
        
        Returns:
            Output tensor of shape (batch_size, k) where k is number of groups
            
        Note:
            If num_features is not divisible by k, input will be zero-padded.
            A warning will be generated if padding occurs, as this typically
            indicates a configuration mismatch.
        """
        # Validate input dimensions
        self._validate_input_dim(x)
        
        # x shape: (batch_size, num_features)
        
        # Optionally permute
        if self.use_randperm:
            perm = torch.randperm(x.shape[-1], device=x.device)
            x = x[:, perm]
        
        # Check if padding is needed and warn
        num_features = x.shape[-1]
        if num_features % self.k != 0:
            pad_size = self.k - (num_features % self.k)
            warnings.warn(
                f"GroupSum: Input has {num_features} features which is not divisible by k={self.k}. "
                f"Adding {pad_size} zero-padded features. "
                f"Expected num_features to be a multiple of {self.k}. "
                f"Check that Layer output_size * num_output_per_node == {self.k} * N for some integer N, "
                f"or adjust k={self.k} to divide {num_features} evenly.",
                UserWarning,
                stacklevel=2
            )
            x = nn.functional.pad(x, (0, pad_size))
        
        # Reshape to (batch_size, k groups, elements_per_group)
        group_size = x.shape[-1] // self.k
        x = x.view(x.shape[0], self.k, group_size)
        
        # Sum within each group and divide by tau
        # Result shape: (batch_size, k)
        return x.sum(dim=-1) / self.tau
