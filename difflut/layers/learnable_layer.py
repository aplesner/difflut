import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type
import warnings
from .base_layer import BaseLUTLayer
from ..registry import register_layer
from ..nodes.node_config import NodeKwargs


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
        
        # Cache for hard indices (computed once when switching to eval mode)
        self.register_buffer('_cached_hard_indices', None)
        self._cache_valid = False
    
    def train(self, mode: bool = True):
        """Override train() to invalidate cache when switching modes."""
        was_training = self.training
        super().train(mode)
        
        # Invalidate cache when switching from eval to train
        if mode and not was_training:
            self._cache_valid = False
        
        return self
    
    def _compute_hard_indices(self) -> torch.Tensor:
        """Compute hard indices from current weights."""
        with torch.no_grad():
            return torch.argmax(self.W, dim=-1)  # (output_size,)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Soft selection (training) or hard selection (eval).
        """
        if self.training:
            # Soft selection - no caching needed
            weights = F.softmax(self.W / self.tau, dim=-1)
            output = torch.matmul(x, weights.t())
        else:
            # Hard selection (evaluation mode)
            # OPTIMIZATION: Cache hard indices to avoid repeated argmax computation
            if not self._cache_valid or self._cached_hard_indices is None:
                self._cached_hard_indices = self._compute_hard_indices()
                self._cache_valid = True
            
            # MEMORY OPTIMIZATION: Use torch.index_select for efficient gathering
            # x: (batch_size, input_size)
            # _cached_hard_indices: (output_size,) with values in range [0, input_size)
            # We want: output[b, o] = x[b, _cached_hard_indices[o]]
            output = torch.index_select(x, 1, self._cached_hard_indices)  # (batch_size, output_size)
        
        return output


@register_layer("learnable")
class LearnableLayer(BaseLUTLayer):
    """
    LUT layer with learnable mapping using nodes.
    Uses soft selection during training and hard selection during evaluation.
    
    The learnable mapping uses a weight matrix W and applies softmax for soft
    selection during training, or argmax for hard selection during evaluation.
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int, 
                 node_type: Type[nn.Module],
                 node_kwargs: NodeKwargs = None,
                 tau: float = 0.001,
                 tau_start: float = 1.0,
                 tau_min: float = 0.0001,
                 tau_decay_iters: float = 1000.0,
                 flip_probability: float = 0.0,
                 grad_stabilization: str = 'none',
                 grad_target_std: float = 1.0,
                 grad_subtract_mean: bool = False,
                 grad_epsilon: float = 1e-8):
        """
        Args:
            input_size: Size of input vector (from encoder or previous layer)
                       Should match: (batch_size, input_size)
            output_size: Number of LUT nodes (output will be batch_size, output_size * output_dim)
            node_type: LUT node class
            node_kwargs: Node configuration (NodeConfig instance or dict with input_dim, output_dim, etc.)
                        Dimension spec: nodes expect (batch_size, output_size, node_input_dim)
            tau: Initial temperature for softmax in learnable mapping
            tau_start: Starting value for tau (used for exponential decay)
            tau_min: Minimum value tau can decay to
            tau_decay_iters: Number of iterations for tau to decay by factor of 10
            flip_probability: Probability of flipping each bit during training (0.0 to 1.0)
            grad_stabilization: Gradient stabilization mode ('none', 'layerwise', 'batchwise')
            grad_target_std: Target standard deviation for gradient rescaling
            grad_subtract_mean: Whether to subtract mean before rescaling
            grad_epsilon: Small constant for numerical stability
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
        super().__init__(input_size, output_size, node_type, node_kwargs, flip_probability,
                        grad_stabilization, grad_target_std, grad_subtract_mean, grad_epsilon)
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through learnable mapping and node.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output tensor of shape (batch_size, output_size * output_dim)
        """
        # Validate input dimensions
        self._validate_input_dims(x)
        
        # Standard approach: get mapping then forward through node
        # PyTorch handles gradients automatically
        mapped_inputs = self.get_mapping(x)
        output = self.node(mapped_inputs)
        
        # Output shape: (batch_size, output_size, output_dim)
        # Reshape to 2D for next layer: (batch_size, output_size * output_dim)
        batch_size = output.shape[0]
        output = output.view(batch_size, -1)
        
        return output
    
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
        
        # Invalidate cache since tau change might affect hard selection
        # (though in practice, argmax is deterministic regardless of tau)
        self.mapping._cache_valid = False

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        flip_str = f", flip_prob={self.flip_probability}" if self.flip_probability > 0 else ""
        grad_str = f", grad_stab={self.grad_stabilization}" if self.grad_stabilization != 'none' else ""
        return f"input_size={self.input_size}, output_size={self.output_size}, " \
               f"n={self.n}, tau={self.tau}, mapping=learnable{flip_str}{grad_str}"