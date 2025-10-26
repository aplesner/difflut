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
    Handles 3D tensors with per-layer-node parameters.
    """
    @staticmethod
    def forward(ctx, input, raw_weights, temperature):
        """
        Forward pass using CUDA kernel.
        
        Args:
            input: (batch_size, layer_size, input_dim) float tensor in [0, 1]
            raw_weights: (layer_size, 2^input_dim, output_dim) float tensor - raw LUT weights (before sigmoid)
            temperature: scalar float
        
        Returns:
            output: (batch_size, layer_size, output_dim) float tensor
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Please compile probabilistic_cuda extension.")
        
        # Ensure correct dtypes and contiguity
        input = input.contiguous().float()
        raw_weights = raw_weights.contiguous().float()
        
        # Call CUDA forward kernel
        output = _probabilistic_cuda_module.forward(input, raw_weights, temperature)
        
        # Save for backward
        ctx.save_for_backward(input, raw_weights)
        ctx.temperature = temperature
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using CUDA kernel.
        
        Args:
            grad_output: (batch_size, layer_size, output_dim) gradient tensor
        
        Returns:
            Gradients for (input, raw_weights, temperature)
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Please compile probabilistic_cuda extension.")
        
        input, raw_weights = ctx.saved_tensors
        temperature = ctx.temperature
        
        # Ensure contiguity
        grad_output = grad_output.contiguous().float()
        
        # Call CUDA backward kernel
        grad_input, grad_weights = _probabilistic_cuda_module.backward(
            input, raw_weights, temperature, grad_output
        )
        
        # Return gradients (None for temperature as it doesn't need gradient)
        return grad_input, grad_weights, None


def probabilistic_forward(input, raw_weights, temperature):
    """
    Probabilistic forward pass with automatic differentiation support.
    
    Args:
        input: (batch_size, layer_size, input_dim) tensor in [0, 1]
        raw_weights: (layer_size, 2^input_dim, output_dim) tensor - raw weights before sigmoid
        temperature: scalar float
    
    Returns:
        output: (batch_size, layer_size, output_dim) tensor
    """
    if _CUDA_EXT_AVAILABLE and input.is_cuda:
        return ProbabilisticFunction.apply(input, raw_weights, temperature)
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
                 layer_size: int = None,
                 init_fn: Optional[Callable] = None,
                 init_kwargs: dict = None,
                 regularizers: dict = None,
                 temperature: float = 1.0,
                 eval_mode: str = "expectation",
                 use_cuda: bool = True):
        """
        Args:
            input_dim: Number of inputs (e.g., 6)
            output_dim: Number of outputs (e.g., 1)
            layer_size: Number of parallel nodes in the layer (for per-layer-node parameters)
            init_fn: Optional initialization function. Should take (param: torch.Tensor, **kwargs)
            init_kwargs: Keyword arguments for init_fn
            regularizers: Dict of custom regularization functions
            temperature: Temperature for sigmoid scaling
            eval_mode: Evaluation mode
            use_cuda: Whether to use CUDA kernels (if available)
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, layer_size=layer_size, 
                         regularizers=regularizers, init_fn=init_fn, init_kwargs=init_kwargs)
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
        
        # Store raw weights (logits) with per-layer-node parameters
        # Shape: (layer_size, 2**input_dim, output_dim)
        self.raw_weights = nn.Parameter(torch.randn(self.layer_size, 2**self.input_dim, self.output_dim))
        self._apply_init_fn(self.raw_weights, name="raw_weights")
        
        # Precompute all binary combinations (LSB-first order) - for CPU fallback
        binary_combinations = []
        for i in range(2**self.input_dim):
            bits = [((i >> j) & 1) for j in range(self.input_dim)]  # LSB first
            binary_combinations.append(bits)
        self.register_buffer('binary_combinations',
                            torch.tensor(binary_combinations, dtype=torch.float32))
        
        # Removed mapping tensor - now using dense connectivity with 3D tensors

    @property
    def weights(self) -> torch.Tensor:
        """Get weights with sigmoid applied (temperature scaled) in [0,1]."""
        return torch.sigmoid(self.raw_weights / self.temperature.clamp(min=1e-6))

    def _binary_to_index(self, x_binary: torch.Tensor) -> torch.Tensor:
        """Convert binary vector to LUT index (LSB-first order)"""
        powers = 2 ** torch.arange(self.input_dim, device=x_binary.device, dtype=torch.float32)
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
        
        Args:
            x: Input tensor (batch_size, layer_size, input_dim)
        Returns:
            Output tensor (batch_size, layer_size, output_dim)
        """
        batch_size, layer_size, input_dim = x.shape
        
        # Try CUDA kernel first (handles 3D tensors directly)
        if self.use_cuda and x.is_cuda and _CUDA_EXT_AVAILABLE:
            # raw_weights is already (layer_size, 2^input_dim, output_dim)
            # Pass temperature as tensor
            output = probabilistic_forward(x, self.raw_weights, self.temperature)
            if output is not None:
                return output
        
        # CPU fallback - process layer by layer with per-layer-node weights
        # weights shape: (layer_size, 2^input_dim, output_dim)
        weights = self.weights
        
        # Ensure binary_combinations is on the same device and dtype as x
        binary_combinations = self.binary_combinations.to(device=x.device, dtype=x.dtype)
        
        # Process each layer independently
        outputs = []
        for layer_idx in range(layer_size):
            x_layer = x[:, layer_idx, :]  # (batch_size, input_dim)
            
            # Vectorized probability computation for this layer
            x_expanded = x_layer.unsqueeze(1)  # (batch_size, 1, input_dim)
            a_expanded = binary_combinations.unsqueeze(0)  # (1, 2^input_dim, input_dim)
            prob_terms = x_expanded * a_expanded + (1 - x_expanded) * (1 - a_expanded)
            probs = torch.prod(prob_terms, dim=-1)  # (batch_size, 2^input_dim)
            
            # Apply per-layer-node weights: (batch_size, 2^input_dim) @ (2^input_dim, output_dim) -> (batch_size, output_dim)
            output_layer = torch.matmul(probs, weights[layer_idx])  # (batch_size, output_dim)
            outputs.append(output_layer)
        
        # Stack outputs: (batch_size, layer_size, output_dim)
        output = torch.stack(outputs, dim=1)
        return output

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Direct LUT lookup simulating hardware behavior.
        Assumes inputs are already binary (0 or 1) from encoder or previous nodes.
        Returns binary outputs (0 or 1).
        
        Args:
            x: Input tensor (batch_size, layer_size, input_dim)
        Returns:
            Output tensor (batch_size, layer_size, output_dim)
        """
        batch_size, layer_size, input_dim = x.shape
        
        # Convert binary inputs to LUT indices (LSB-first order)
        powers = 2 ** torch.arange(input_dim, device=x.device, dtype=x.dtype)
        indices = (x * powers).sum(dim=-1).long()  # (batch_size, layer_size)
        
        # Look up per-layer-node weights and threshold at 0.5 to get binary output
        weights = self.weights  # (layer_size, 2^input_dim, output_dim)
        
        # Gather per-layer-node weights
        outputs = []
        for layer_idx in range(layer_size):
            batch_indices = indices[:, layer_idx]  # (batch_size,)
            output_layer = weights[layer_idx][batch_indices]  # (batch_size, output_dim)
            outputs.append(output_layer)
        
        # Stack outputs: (batch_size, layer_size, output_dim)
        output = torch.stack(outputs, dim=1)
        
        # Threshold to get binary output (weights are in [0, 1] after sigmoid)
        output = (output >= 0.5).float()
        
        return output

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization by default."""
        return torch.tensor(0.0, device=self.raw_weights.device)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"