import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Dict, Any, Optional
import warnings
from .base_layer import BaseLUTLayer
from ..registry import register_layer


class LearnableMappingModule(nn.Module):
    """
    Helper module for learnable mapping (not registered, used internally).
    Provides soft selection during training and hard selection during evaluation.
    """
    
    def __init__(self, input_size: int, output_size: int, tau: float = 0.001):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tau = tau
        
        # Weight matrix
        self.W = nn.Parameter(torch.randn(output_size, input_size))
        nn.init.xavier_uniform_(self.W)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Soft selection (training) or hard selection (eval).
        """
        if self.training:
            # Soft selection
            weights = F.softmax(self.W / self.tau, dim=-1)
            output = torch.matmul(x, weights.t())
        else:
            # Hard selection (evaluation mode)
            # MEMORY OPTIMIZATION: Avoid expanding (batch, output_size, input_size) tensor
            hard_indices = torch.argmax(self.W, dim=-1)  # (output_size,)
            
            # Use advanced indexing to gather without creating huge expanded tensor
            batch_size = x.shape[0]
            batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1)  # (batch, 1)
            batch_indices = batch_indices.expand(-1, self.output_size)  # (batch, output_size)
            
            # Expand hard_indices for batch dimension
            hard_indices_expanded = hard_indices.unsqueeze(0).expand(batch_size, -1)  # (batch, output_size)
            
            # Advanced indexing: x[batch_indices, hard_indices]
            output = x[batch_indices, hard_indices_expanded]  # (batch, output_size)
        
        return output


@register_layer("learnable")
class LearnableLayer(BaseLUTLayer):
    """
    LUT layer with learnable mapping using nodes.
    Uses soft selection during training and hard selection during evaluation.
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int, 
                 node_type: Type[nn.Module],
                 node_kwargs: Optional[Dict[str, Any]] = None,
                 tau: float = 0.001,
                 tau_start: float = 1.0,
                 tau_min: float = 0.0001,
                 tau_decay_iters: float = 1000.0):
        """
        Args:
            input_size: Size of input vector (from encoder or previous layer)
                       Should match: (batch_size, input_size)
            output_size: Number of LUT nodes (output will be batch_size, output_size * output_dim)
            node_type: LUT node class
            node_kwargs: Additional arguments for nodes (should include input_dim and output_dim)
                        Dimension spec: nodes expect (batch_size, output_size, node_input_dim)
            tau: Initial temperature for softmax in learnable mapping
            tau_start: Starting value for tau (used for exponential decay)
            tau_min: Minimum value tau can decay to
            tau_decay_iters: Number of iterations for tau to decay by factor of 10
        """
        # Warn if tau parameters seem unusual
        if tau_start < tau_min:
            warnings.warn(
                f"LearnableLayer: tau_start ({tau_start}) is less than tau_min ({tau_min}). "
                f"This means tau will be clamped immediately. Set tau_start >= tau_min.",
                UserWarning,
                stacklevel=2
            )
        
        # Initialize parent with nodes (n will be extracted from created nodes)
        super().__init__(input_size, output_size, node_type, node_kwargs)
        
        # Warn about parameter count after n is known
        total_connections = output_size * self.n
        if total_connections > input_size * 10:
            warnings.warn(
                f"LearnableLayer: Creating {total_connections} learnable connections from {input_size} inputs. "
                f"This may lead to overfitting. Consider using GroupedLayer or fewer nodes/inputs per node (n={self.n}).",
                UserWarning,
                stacklevel=2
            )
        
        # Tau decay parameters
        self.tau_start = tau_start
        self.tau_min = tau_min
        self.tau_decay_iters = tau_decay_iters
        self.tau = tau_start  # Start with tau_start instead of tau
        
        # Create learnable mapping module (helper, not registered)
        # Note: self.n is now available from parent's __init__
        self.mapping = LearnableMappingModule(input_size, output_size * self.n, self.tau)
    
    def get_mapping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable mapping and reshape for nodes.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        Returns:
            Mapped inputs of shape (batch_size, output_size, n)
        """
        batch_size = x.shape[0]
        
        # Apply learnable mapping
        mapped_flat = self.mapping(x)  # (batch_size, output_size * n)
        
        # Reshape for nodes
        mapped_inputs = mapped_flat.view(batch_size, self.output_size, self.n)
        
        return mapped_inputs
    
    def get_mapping_matrix(self) -> torch.Tensor:
        """Get current hard mapping (for inspection)."""
        with torch.no_grad():
            hard_indices = torch.argmax(self.mapping.W, dim=-1)
            return hard_indices.view(self.output_size, self.n)
    
    def update_tau(self, iteration: int):
        """
        Update tau using exponential decay.
        
        Args:
            iteration: Current training iteration
        """
        # Calculate decay factor: tau = tau_start * 10^(-iteration / tau_decay_iters)
        # This means tau decays by a factor of 10 every tau_decay_iters iterations
        decay_factor = 10.0 ** (-iteration / self.tau_decay_iters)
        self.tau = max(self.tau_start * decay_factor, self.tau_min)
        
        # Update the mapping module's tau
        self.mapping.tau = self.tau

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return f"input_size={self.input_size}, output_size={self.output_size}, " \
               f"n={self.n}, tau={self.tau}, mapping=learnable"