import torch
import torch.nn as nn
import itertools
from typing import Optional, Callable
from .base_node import BaseNode
from ..registry import register_node

@register_node("polylut")
class PolyLUTNode(BaseNode):
    """
    Polynomial LUT Node that computes multivariate polynomials up to degree D.
    Uses autograd for gradient computation.
    """
    
    def __init__(self, 
                 input_dim: int = None,
                 output_dim: int = None,
                 degree: int = 2,
                 init_fn: Optional[Callable] = None,
                 init_kwargs: dict = None,
                 regularizers: dict = None):
        """
        Args:
            input_dim: Number of inputs (e.g., 6)
            output_dim: Number of outputs (e.g., 1)
            degree: Maximum degree of polynomial terms
            init_fn: Optional initialization function. Should take (param: torch.Tensor, **kwargs)
            init_kwargs: Keyword arguments for init_fn
            regularizers: Dict of custom regularization functions
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, regularizers=regularizers, init_fn=init_fn, init_kwargs=init_kwargs)
        self.degree = degree
        
        # Generate all monomial combinations up to degree D
        self.monomial_combinations = self._generate_monomial_combinations(self.num_inputs, degree)
        self.num_monomials = len(self.monomial_combinations)
        
        # Store as buffer for efficient computation
        self.register_buffer('exponent_matrix', 
                           torch.tensor(self.monomial_combinations, dtype=torch.float32))
        
        # Initialize weights for polynomial coefficients with default values, then apply init_fn if provided
        self.weights = nn.Parameter(torch.randn(self.num_monomials, self.num_outputs) * 0.1)
        self._apply_init_fn(self.weights, name="weights")

    def _generate_monomial_combinations(self, num_inputs: int, degree: int) -> list:
        """Generate all monomial combinations up to given degree."""
        combinations = []
        
        # Include constant term
        combinations.append(tuple([0] * num_inputs))
        
        # Generate monomials for degrees 1 to D
        for d in range(1, degree + 1):
            for exponents in self._integer_compositions(num_inputs, d):
                combinations.append(exponents)
        
        return combinations

    def _integer_compositions(self, n: int, d: int):
        """Generate all n-tuples of non-negative integers that sum to d."""
        if n == 1:
            yield (d,)
        else:
            for i in range(d + 1):
                for comp in self._integer_compositions(n - 1, d - i):
                    yield (i,) + comp

    def _compute_monomials(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute all monomial values for input x.
        Args:
            x: shape [batch_size, num_inputs] with values in [0,1]
        Returns:
            monomial matrix of shape [batch_size, num_monomials]
        """
        # Vectorized monomial computation
        x_expanded = x.unsqueeze(1)  # [batch, 1, num_inputs]
        exponents = self.exponent_matrix.unsqueeze(0)  # [1, num_monomials, num_inputs]
        
        # Handle zero exponents (avoid 0^0)
        mask = exponents > 0
        safe_x = torch.where(mask, x_expanded, torch.ones_like(x_expanded))
        
        # Compute x^exponents
        monomials = (safe_x ** exponents).prod(dim=-1)  # [batch, num_monomials]
        
        return monomials

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training: polynomial transform + sigmoid activation.
        
        Args:
            x: Input tensor (batch_size, layer_size, num_inputs)
        Returns:
            Output tensor (batch_size, layer_size, num_outputs)
        """
        batch_size, layer_size, input_dim = x.shape
        # Reshape to (batch_size * layer_size, num_inputs)
        x_flat = x.view(batch_size * layer_size, input_dim)
        
        # Compute all monomials
        monomials = self._compute_monomials(x_flat)  # (batch_size * layer_size, num_monomials)
        
        # Linear combination of monomials
        z = torch.matmul(monomials, self.weights)  # (batch_size * layer_size, num_outputs)
        output_flat = torch.sigmoid(z)
        
        # Ensure output is always 2D
        if output_flat.dim() == 1:
            output_flat = output_flat.unsqueeze(1)  # (batch_size * layer_size, 1)
        
        # Reshape back to (batch_size, layer_size, num_outputs)
        output = output_flat.view(batch_size, layer_size, self.num_outputs)
        return output

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Discretize by applying Heaviside at 0.5 to forward_train output.
        This makes it behave like a real LUT with binary outputs.
        
        Args:
            x: Input tensor (batch_size, layer_size, num_inputs)
        Returns:
            Output tensor (batch_size, layer_size, num_outputs)
        """
        batch_size, layer_size, input_dim = x.shape
        # Reshape to (batch_size * layer_size, num_inputs)
        x_flat = x.view(batch_size * layer_size, input_dim)
        
        # Compute same as forward_train (polynomial + sigmoid)
        monomials = self._compute_monomials(x_flat)
        z = torch.matmul(monomials, self.weights)
        output_flat = (z >= 0.0).float()
        
        # Ensure output is always 2D
        if output_flat.dim() == 1:
            output_flat = output_flat.unsqueeze(1)  # (batch_size * layer_size, 1)
        
        # Reshape back to (batch_size, layer_size, num_outputs)
        output = output_flat.view(batch_size, layer_size, self.num_outputs)
        return output

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization by default."""
        return torch.tensor(0.0, device=self.weights.device)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, " \
               f"degree={self.degree}, num_monomials={self.num_monomials}"