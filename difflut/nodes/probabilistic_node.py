import torch
import torch.nn as nn
import itertools
from typing import Optional, Callable
from .base_node import BaseNode
from ..registry import register_node

@register_node("probabilistic")
class ProbabilisticNode(BaseNode):
    """
    Probabilistic LUT node with continuous inputs in [0,1].
    Uses probabilistic forward pass and autograd for gradients.
    """
    
    def __init__(self, 
                 input_dim: list = None,
                 output_dim: list = None,
                 init_fn: Optional[Callable] = None,
                 regularizers: dict = None,
                 temperature: float = 1.0,
                 eval_mode: str = "expectation"):
        """
        Args:
            input_dim: Input dimensions as list (e.g., [6])
            output_dim: Output dimensions as list (e.g., [1])
            init_fn: Optional initialization function
            regularizers: Dict of custom regularization functions
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, regularizers=regularizers)
        self.register_buffer('temperature', torch.tensor(float(temperature)))
        assert eval_mode in {"expectation", "deterministic", "threshold"}, "Invalid eval_mode"
        self.eval_mode = eval_mode
        
        # Store raw weights (logits) that will be passed through sigmoid
        if init_fn:
            self.raw_weights = nn.Parameter(init_fn((2**self.num_inputs, self.num_outputs)))
        else:
            self.raw_weights = nn.Parameter(torch.randn(2**self.num_inputs, self.num_outputs))
        
        # Precompute all binary combinations (MSB-first order)
        binary_combinations = []
        for i in range(2**self.num_inputs):
            bits = [((i >> j) & 1) for j in reversed(range(self.num_inputs))]  # MSB first
            binary_combinations.append(bits)
        self.register_buffer('binary_combinations',
                            torch.tensor(binary_combinations, dtype=torch.float32))

    @property
    def weights(self) -> torch.Tensor:
        """Get weights with sigmoid applied (temperature scaled) in [0,1]."""
        return torch.sigmoid(self.raw_weights / self.temperature.clamp(min=1e-6))

    def _binary_to_index(self, x_binary: torch.Tensor) -> torch.Tensor:
        """Convert binary vector to LUT index (MSB-first order)"""
        powers = 2 ** torch.arange(self.num_inputs - 1, -1, -1, device=x_binary.device, dtype=torch.float32)
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
        # Ensure binary_combinations is on the same device and dtype as x
        binary_combinations = self.binary_combinations.to(device=x.device, dtype=x.dtype)
        # Vectorized probability computation
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, num_inputs)
        a_expanded = binary_combinations.unsqueeze(0)  # (1, 2^num_inputs, num_inputs)
        prob_terms = x_expanded * a_expanded + (1 - x_expanded) * (1 - a_expanded)
        probs = torch.prod(prob_terms, dim=2)
        weights = self.weights
        output = torch.matmul(probs, weights)
        if self.num_outputs == 1:
            output = output.squeeze(-1)
        return output

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Discretize by applying Heaviside at 0.5 to forward_train output.
        This makes it behave like a real LUT with binary outputs.
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # Compute same as forward_train (probabilistic expectation)
        batch_size = x.shape[0]
        # Ensure binary_combinations is on the same device and dtype as x
        binary_combinations = self.binary_combinations.to(device=x.device, dtype=x.dtype)
        # Vectorized probability computation
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, num_inputs)
        a_expanded = binary_combinations.unsqueeze(0)  # (1, 2^num_inputs, num_inputs)
        prob_terms = x_expanded * a_expanded + (1 - x_expanded) * (1 - a_expanded)
        probs = torch.prod(prob_terms, dim=2)  # (batch_size, 2^num_inputs)
        weights = self.weights
        output = torch.matmul(probs, weights)
        # Discretize: Heaviside at 0.5 since forward_train output is in [0,1]
        output = (output >= 0.5).float()
        if self.num_outputs == 1:
            output = output.squeeze(-1)
        return output

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization by default."""
        return torch.tensor(0.0, device=self.raw_weights.device)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"