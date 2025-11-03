import torch
import torch.nn as nn
import itertools
from typing import Optional, Callable, Dict, Any, Tuple, List
from .base_node import BaseNode
from ..registry import register_node

# Default maximum polynomial degree for PolyLUT nodes
DEFAULT_POLYLUT_DEGREE: int = 3

@register_node("polylut")
class PolyLUTNode(BaseNode):
    """
    Polynomial LUT Node that computes multivariate polynomials up to degree D.
    Uses autograd for gradient computation.
    Now supports per-layer-node coefficients for better memory access patterns.
    """
    
    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        layer_size: Optional[int] = None,
        degree: int = DEFAULT_POLYLUT_DEGREE,
        init_fn: Optional[Callable[[torch.Tensor], None]] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        regularizers: Optional[Dict[str, Tuple[Callable, float, Dict[str, Any]]]] = None
    ) -> None:
        """
        Args:
            input_dim: Number of inputs (e.g., 6)
            output_dim: Number of outputs (e.g., 1)
            layer_size: Number of parallel nodes in the layer (e.g., 128)
            degree: Maximum degree of polynomial terms
            init_fn: Optional initialization function. Should take (param: torch.Tensor, **kwargs)
            init_kwargs: Keyword arguments for init_fn
            regularizers: Dict of custom regularization functions
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, layer_size=layer_size,
                         regularizers=regularizers, init_fn=init_fn, init_kwargs=init_kwargs)
        self.degree = degree
        
        # Generate all monomial combinations up to degree D
        self.monomial_combinations = self._generate_monomial_combinations(self.num_inputs, degree)
        self.num_monomials = len(self.monomial_combinations)
        
        # Store as buffer for efficient computation
        self.register_buffer('exponent_matrix', 
                           torch.tensor(self.monomial_combinations, dtype=torch.float32))
        
        # Initialize weights for polynomial coefficients with per-layer-node parameters
        # Shape: (layer_size, num_monomials, num_outputs)
        self.weights = nn.Parameter(torch.randn(self.layer_size, self.num_monomials, self.num_outputs) * 0.1)
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
            x: shape [batch_size, layer_size, num_inputs] with values in [0,1]
        Returns:
            monomial matrix of shape [batch_size, layer_size, num_monomials]
        """
        # Vectorized monomial computation
        x_expanded = x.unsqueeze(2)  # [batch, layer_size, 1, num_inputs]
        exponents = self.exponent_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, num_monomials, num_inputs]
        
        # Handle zero exponents (avoid 0^0)
        mask = exponents > 0
        safe_x = torch.where(mask, x_expanded, torch.ones_like(x_expanded))
        
        # Compute x^exponents
        monomials = (safe_x ** exponents).prod(dim=-1)  # [batch, layer_size, num_monomials]
        
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
        
        # Verify layer_size matches
        if layer_size != self.layer_size:
            raise ValueError(
                f"Input layer_size {layer_size} does not match node's layer_size {self.layer_size}"
            )
        
        # Compute all monomials: (batch_size, layer_size, num_monomials)
        monomials = self._compute_monomials(x)
        
        # Linear combination of monomials with per-layer-node weights
        # monomials: (batch_size, layer_size, num_monomials)
        # weights: (layer_size, num_monomials, num_outputs)
        # output: (batch_size, layer_size, num_outputs)
        z = torch.einsum('blm,lmo->blo', monomials, self.weights)
        output = torch.sigmoid(z)
        
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
        
        # Verify layer_size matches
        if layer_size != self.layer_size:
            raise ValueError(
                f"Input layer_size {layer_size} does not match node's layer_size {self.layer_size}"
            )
        
        # Compute all monomials: (batch_size, layer_size, num_monomials)
        monomials = self._compute_monomials(x)
        
        # Linear combination with per-layer-node weights
        z = torch.einsum('blm,lmo->blo', monomials, self.weights)
        
        # Discretize at 0.0
        output = (z >= 0.0).float()
        
        return output

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization by default."""
        return torch.tensor(0.0, device=self.weights.device)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, " \
               f"degree={self.degree}, num_monomials={self.num_monomials}"