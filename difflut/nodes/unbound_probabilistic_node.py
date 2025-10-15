import torch
import torch.nn as nn
import itertools
from typing import Optional, Callable
from .base_node import BaseNode
from ..registry import register_node
    
@register_node("unbound_probabilistic")
class UnboundProbabilisticNode(BaseNode):
    """
    Unbound Probabilistic LUT node with continuous inputs in [0,1].
    Uses probabilistic forward pass and autograd for gradients.
    """
    
    def __init__(self, 
                 num_inputs: int,
                 output_dim: int = 1,
                 init_fn: Optional[Callable] = None,
                 regularizers: dict = None):
        """
        Args:
            num_inputs: Number of LUT inputs
            output_dim: Number of output bits
            init_fn: Optional initialization function
            regularizers: Dict of custom regularization functions
        """
        super().__init__(num_inputs=num_inputs, regularizers=regularizers)
        self.output_dim = output_dim
        
        # Store raw weights (logits) that will be passed through sigmoid
        if init_fn:
            self.weights = nn.Parameter(init_fn((2**num_inputs, output_dim)))
        else:
            self.weights = nn.Parameter(torch.randn(2**num_inputs, output_dim))
        
        # Precompute all binary combinations
        binary_combinations = []
        for i in range(2**num_inputs):
            bits = []
            for j in range(num_inputs):
                bits.append((i >> j) & 1)
            binary_combinations.append(bits)
        
        self.register_buffer('binary_combinations',
                           torch.tensor(binary_combinations, dtype=torch.float32))


    def _binary_to_index(self, x_binary: torch.Tensor) -> torch.Tensor:
        """Convert binary vector to LUT index"""
        powers = 2 ** torch.arange(self.num_inputs - 1, -1, -1, 
                                   device=x_binary.device, dtype=torch.float32)
        if x_binary.dim() == 1:
            return (x_binary * powers).sum().long()
        else:
            return (x_binary * powers).sum(dim=-1).long()

    def _bernoulli_probability(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Compute Pr(a|x) = Π_j [x_j^a_j * (1-x_j)^(1-a_j)]
        Args:
            x: continuous inputs in [0,1], shape (batch_size, num_inputs)
            a: binary pattern, shape (num_inputs,) or (batch_size, num_inputs)
        """
        if a.dim() == 1:
            a = a.unsqueeze(0).expand(x.shape[0], -1)
        
        # Compute probability
        prob = x * a + (1 - x) * (1 - a)
        return torch.prod(prob, dim=1)

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training: probabilistic expectation (vectorized)
        f(x) = Σ_a ω_δ(a) * Pr(a|x)
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        
        batch_size = x.shape[0]
        
        # Vectorized probability computation
        # x: (batch_size, num_inputs)
        # binary_combinations: (2^num_inputs, num_inputs)
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, num_inputs)
        a_expanded = self.binary_combinations.unsqueeze(0)  # (1, 2^num_inputs, num_inputs)
        
        # Compute Pr(a|x) for all combinations at once
        # prob_terms = x^a * (1-x)^(1-a)
        prob_terms = x_expanded * a_expanded + (1 - x_expanded) * (1 - a_expanded)
        # Product over input dimensions: (batch_size, 2^num_inputs)
        probs = torch.prod(prob_terms, dim=2)
        
             
        # Matrix multiply: (batch_size, 2^num_inputs) @ (2^num_inputs, output_dim) -> (batch_size, output_dim)
        output = torch.sigmoid(torch.matmul(probs, self.weights))
        
        if self.output_dim == 1:
            output = output.squeeze(-1)
        
        return output

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization by default."""
        return torch.tensor(0.0, device=self.weights.device)

    def extra_repr(self) -> str:
        return f"num_inputs={self.num_inputs}, output_dim={self.output_dim}"