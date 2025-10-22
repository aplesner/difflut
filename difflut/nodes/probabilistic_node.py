import torch
import torch.nn as nn
import itertools
from typing import Optional, Callable
import warnings
from .base_node import BaseNode
from ..registry import register_node
from .cuda import is_cuda_available

# Try to import the compiled CUDA extension
try:
    import probabilistic_cuda as _probabilistic_cuda_module
    _CUDA_EXT_AVAILABLE = True
except ImportError:
    _CUDA_EXT_AVAILABLE = False
    _probabilistic_cuda_module = None
    warnings.warn(
        "CUDA extension 'probabilistic_cuda' not available. ProbabilisticNode will use slower CPU fallback. "
        "For better performance, compile the CUDA extension using: "
        "'cd difflut && python setup.py install'. "
        "To suppress this warning: warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut.nodes.probabilistic_node')",
        RuntimeWarning,
        stacklevel=2
    )


class ProbabilisticFunction(torch.autograd.Function):
    """
    PyTorch autograd function wrapper for Probabilistic CUDA kernels.
    """
    @staticmethod
    def forward(ctx, input, mapping, luts, temperature):
        """
        Forward pass using CUDA kernel.
        
        Args:
            input: (batch_size, input_length) float tensor in [0, 1]
            mapping: (num_luts, n) int tensor - mapping of inputs to each LUT
            luts: (num_luts, 2^n) float tensor - raw LUT weights (before sigmoid)
            temperature: scalar tensor
        
        Returns:
            output: (batch_size, num_luts) float tensor
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Please compile probabilistic_cuda extension.")
        
        # Ensure correct dtypes and contiguity
        input = input.contiguous().float()
        mapping = mapping.contiguous().int()
        luts = luts.contiguous().float()
        
        # Call CUDA forward kernel
        output = _probabilistic_cuda_module.forward(input, mapping, luts, temperature)
        
        # Save for backward
        ctx.save_for_backward(input, mapping, luts, temperature)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using CUDA kernel.
        
        Args:
            grad_output: (batch_size, num_luts) gradient tensor
        
        Returns:
            Gradients for (input, mapping, luts, temperature)
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Please compile probabilistic_cuda extension.")
        
        input, mapping, luts, temperature = ctx.saved_tensors
        
        # Ensure contiguity
        grad_output = grad_output.contiguous().float()
        
        # Call CUDA backward kernel
        grad_input, grad_luts = _probabilistic_cuda_module.backward(
            input, mapping, luts, temperature, grad_output
        )
        
        # Return gradients (None for mapping and temperature as they don't need gradients)
        return grad_input, None, grad_luts, None


def probabilistic_forward(input, mapping, luts, temperature):
    """
    Probabilistic forward pass with automatic differentiation support.
    
    Args:
        input: (batch_size, input_length) tensor in [0, 1]
        mapping: (num_luts, n) int tensor
        luts: (num_luts, 2^n) tensor - raw weights before sigmoid
        temperature: scalar tensor
    
    Returns:
        output: (batch_size, num_luts) tensor
    """
    if _CUDA_EXT_AVAILABLE and input.is_cuda:
        return ProbabilisticFunction.apply(input, mapping, luts, temperature)
    else:
        # CPU fallback handled in forward_train
        return None

@register_node("probabilistic")
class ProbabilisticNode(BaseNode):
    """
    Probabilistic LUT node with continuous inputs in [0,1].
    Uses probabilistic forward pass and autograd for gradients.
    """
    
    def __init__(self, 
                 input_dim: int = None,
                 output_dim: int = None,
                 init_fn: Optional[Callable] = None,
                 regularizers: dict = None,
                 temperature: float = 1.0,
                 eval_mode: str = "expectation",
                 use_cuda: bool = True):
        """
        Args:
            input_dim: Number of inputs (e.g., 6)
            output_dim: Number of outputs (e.g., 1)
            init_fn: Optional initialization function
            regularizers: Dict of custom regularization functions
            temperature: Temperature for sigmoid scaling
            eval_mode: Evaluation mode
            use_cuda: Whether to use CUDA kernels (if available)
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, regularizers=regularizers, init_fn=init_fn)
        self.register_buffer('temperature', torch.tensor(float(temperature)))
        assert eval_mode in {"expectation", "deterministic", "threshold"}, "Invalid eval_mode"
        self.eval_mode = eval_mode
        self.use_cuda = use_cuda and is_cuda_available()
        
        # Warn if CUDA requested but not available
        if use_cuda and not _CUDA_EXT_AVAILABLE:
            warnings.warn(
                "ProbabilisticNode: CUDA was requested (use_cuda=True) but CUDA extension is not available. "
                "Using CPU fallback which may be significantly slower. "
                "To enable CUDA: compile the extension with 'cd difflut && python setup.py install'",
                RuntimeWarning,
                stacklevel=2
            )
        
        # Store raw weights (logits) that will be passed through sigmoid
        if self.init_fn:
            self.raw_weights = nn.Parameter(self.init_fn((2**self.num_inputs, self.num_outputs)))
        else:
            self.raw_weights = nn.Parameter(torch.randn(2**self.num_inputs, self.num_outputs))
        
        # Precompute all binary combinations (LSB-first order) - for CPU fallback
        binary_combinations = []
        for i in range(2**self.num_inputs):
            bits = [((i >> j) & 1) for j in range(self.num_inputs)]  # LSB first
            binary_combinations.append(bits)
        self.register_buffer('binary_combinations',
                            torch.tensor(binary_combinations, dtype=torch.float32))
        
        # Create mapping tensor for CUDA (each LUT maps to all inputs in order)
        # Shape: (num_outputs, num_inputs)
        self.register_buffer('mapping', torch.arange(self.num_inputs, dtype=torch.int32).unsqueeze(0).expand(self.num_outputs, -1))

    @property
    def weights(self) -> torch.Tensor:
        """Get weights with sigmoid applied (temperature scaled) in [0,1]."""
        return torch.sigmoid(self.raw_weights / self.temperature.clamp(min=1e-6))

    def _binary_to_index(self, x_binary: torch.Tensor) -> torch.Tensor:
        """Convert binary vector to LUT index (LSB-first order)"""
        powers = 2 ** torch.arange(self.num_inputs, device=x_binary.device, dtype=torch.float32)
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
        Uses CUDA kernel if available, otherwise falls back to CPU.
        """
        # Try CUDA kernel first
        if self.use_cuda and x.is_cuda and _CUDA_EXT_AVAILABLE:
            # Transpose raw_weights to match CUDA kernel expectations
            # raw_weights is (2^n, num_outputs), need (num_outputs, 2^n)
            luts = self.raw_weights.t().contiguous()
            mapping = self.mapping.int()
            
            output = probabilistic_forward(x, mapping, luts, self.temperature)
            return output
        
        # CPU fallback - original vectorized implementation
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
        
        return output

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Direct LUT lookup simulating hardware behavior.
        Assumes inputs are already binary (0 or 1) from encoder or previous nodes.
        Returns binary outputs (0 or 1).
        """
        batch_size = x.shape[0]
        
        # Convert binary inputs to LUT indices (MSB-first)
        # For each sample, compute index = sum(x[i] * 2^(n-1-i))
        indices = self._binary_to_index(x)  # (batch_size,)
        
        # Look up weights and threshold at 0.5 to get binary output
        weights = self.weights  # (2^num_inputs, num_outputs)
        output = weights[indices]  # (batch_size, num_outputs)
        
        # Threshold to get binary output (weights are in [0, 1] after sigmoid)
        output = (output >= 0.5).float()
        
        return output

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization by default."""
        return torch.tensor(0.0, device=self.raw_weights.device)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"