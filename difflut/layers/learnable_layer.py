import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Dict, Any, Optional
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
            # Hard selection
            hard_indices = torch.argmax(self.W, dim=-1)
            # Gather from input
            output = torch.gather(
                x.unsqueeze(1).expand(-1, self.output_size, -1),
                2,
                hard_indices.unsqueeze(0).unsqueeze(2).expand(x.shape[0], -1, 1)
            ).squeeze(2)
        
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
                 n: int = 6,
                 node_kwargs: Optional[Dict[str, Any]] = None,
                 tau: float = 0.001,
                 tau_start: float = 1.0,
                 tau_min: float = 0.0001,
                 tau_decay_iters: float = 1000.0):
        """
        Args:
            input_size: Size of input vector
            output_size: Number of LUT nodes
            node_type: LUT node class
            n: Number of inputs per LUT
            node_kwargs: Additional arguments for nodes
            tau: Initial temperature for softmax in learnable mapping
            tau_start: Starting value for tau (used for exponential decay)
            tau_min: Minimum value tau can decay to
            tau_decay_iters: Number of iterations for tau to decay by factor of 10
        """
        # Initialize parent with nodes
        super().__init__(input_size, output_size, node_type, n, node_kwargs)
        
        # Tau decay parameters
        self.tau_start = tau_start
        self.tau_min = tau_min
        self.tau_decay_iters = tau_decay_iters
        self.tau = tau_start  # Start with tau_start instead of tau
        
        # Create learnable mapping module (helper, not registered)
        self.mapping = LearnableMappingModule(input_size, output_size * n, self.tau)
    
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