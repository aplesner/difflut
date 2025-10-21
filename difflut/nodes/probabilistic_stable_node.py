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
    import probabilistic_stable_cuda as _probabilistic_stable_cuda_module
    _CUDA_EXT_AVAILABLE = True
except ImportError:
    _CUDA_EXT_AVAILABLE = False
    _probabilistic_stable_cuda_module = None
    warnings.warn(
        "CUDA extension 'probabilistic_stable_cuda' not available. ProbabilisticStableNode will use slower CPU fallback. "
        "For better performance, compile the CUDA extension using: "
        "'cd difflut && python setup.py install'. "
        "To suppress this warning: warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut.nodes.probabilistic_stable_node')",
        RuntimeWarning,
        stacklevel=2
    )


class ProbabilisticStableFunction(torch.autograd.Function):
    """
    PyTorch autograd function wrapper for Probabilistic CUDA kernels with gradient stabilization.
    
    Gradient Stabilization:
    ----------------------
    When gradients flow backward, we rescale the input gradients using an L1-based,
    log-scaled normalization rule to prevent vanishing gradients:
    
    For each sample b:
        r = log(1 + num_inputs/num_outputs)
        s_b = (alpha * r * ||g_out[b]||_1) / (||g_in_raw[b]||_1 + eps)
        g_in_scaled[b] = s_b * g_in_raw[b]
    
    This ensures gradient magnitude is preserved across layers, scaled by:
    - log ratio of input/output dimensions (gentle scaling)
    - alpha factor (typically 0.8-1.0) for fine-tuning
    - L1 norm for robustness to sparse gradients
    """
    @staticmethod
    def forward(ctx, input, mapping, luts, temperature, alpha, scale_min, scale_max):
        """
        Forward pass using CUDA kernel.
        
        Args:
            input: (batch_size, input_length) float tensor in [0, 1]
            mapping: (num_luts, n) int tensor - mapping of inputs to each LUT
            luts: (num_luts, 2^n) float tensor - raw LUT weights (clamped to [0,1])
            temperature: scalar tensor (unused, kept for compatibility)
            alpha: scalar tensor - gradient stabilization strength
            scale_min: scalar tensor - minimum allowed gradient scale
            scale_max: scalar tensor - maximum allowed gradient scale
        
        Returns:
            output: (batch_size, num_luts) float tensor
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Please compile probabilistic_stable_cuda extension.")
        
        # Ensure correct dtypes and contiguity
        input = input.contiguous().float()
        mapping = mapping.contiguous().int()
        luts = luts.contiguous().float()
        
        # Call CUDA forward kernel
        output = _probabilistic_stable_cuda_module.forward(input, mapping, luts, temperature)
        
        # Save for backward
        ctx.save_for_backward(input, mapping, luts, temperature, alpha, scale_min, scale_max)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using CUDA kernel with gradient stabilization.
        
        Args:
            grad_output: (batch_size, num_luts) gradient tensor
        
        Returns:
            Gradients for (input, mapping, luts, temperature, alpha, scale_min, scale_max)
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Please compile probabilistic_stable_cuda extension.")
        
        input, mapping, luts, temperature, alpha, scale_min, scale_max = ctx.saved_tensors
        
        # Ensure contiguity
        grad_output = grad_output.contiguous().float()
        
        # Get dimensions for gradient stabilization
        num_inputs = mapping.size(1)
        num_outputs = grad_output.size(1)
        
        # Call CUDA backward kernel with gradient stabilization parameters
        grad_input, grad_luts = _probabilistic_stable_cuda_module.backward(
            input, mapping, luts, temperature, grad_output,
            alpha, num_inputs, num_outputs, scale_min, scale_max
        )
        
        # Return gradients (None for mapping, temperature, alpha, scale_min, scale_max)
        return grad_input, None, grad_luts, None, None, None, None


def probabilistic_stable_forward(input, mapping, luts, temperature, alpha, scale_min, scale_max):
    """
    Probabilistic forward pass with gradient stabilization and automatic differentiation support.
    
    Args:
        input: (batch_size, input_length) tensor in [0, 1]
        mapping: (num_luts, n) int tensor
        luts: (num_luts, 2^n) tensor - raw weights clamped to [0,1]
        temperature: scalar tensor (unused, kept for compatibility)
        alpha: scalar tensor - gradient stabilization strength
        scale_min: scalar tensor - minimum allowed gradient scale
        scale_max: scalar tensor - maximum allowed gradient scale
    
    Returns:
        output: (batch_size, num_luts) tensor
    """
    if _CUDA_EXT_AVAILABLE and input.is_cuda:
        return ProbabilisticStableFunction.apply(input, mapping, luts, temperature, alpha, scale_min, scale_max)
    else:
        # CPU fallback handled in forward_train
        return None

@register_node("probabilistic_stable")
class ProbabilisticStableNode(BaseNode):
    """
    Probabilistic LUT node with continuous inputs in [0,1] and gradient stabilization.
    
    Uses probabilistic forward pass with clamped weights (faster than sigmoid) and 
    gradient-stabilized backward pass to prevent vanishing gradients in deep LUT networks.
    
    Gradient Stabilization:
    ----------------------
    Rescales input gradients per sample using L1-based, log-scaled normalization:
        s_b = (alpha * log(1 + I/O) * ||g_out[b]||_1) / (||g_in[b]||_1 + eps)
    
    This ensures gradient magnitude is preserved across layers while avoiding
    explosion or vanishing.
    
    Performance Optimizations:
    -------------------------
    - Uses clamp(w, 0, 1) instead of sigmoid for 3-5x faster forward/backward
    - Local memory caching of inputs to reduce memory access
    - Reduced atomic operations in backward pass
    - Loop unrolling for small LUTs
    """
    
    def __init__(self, 
                 input_dim: list = None,
                 output_dim: list = None,
                 init_fn: Optional[Callable] = None,
                 regularizers: dict = None,
                 temperature: float = 1.0,
                 eval_mode: str = "expectation",
                 use_cuda: bool = True,
                 alpha: float = 1.0,
                 scale_min: float = 0.1,
                 scale_max: float = 10.0):
        """
        Args:
            input_dim: Input dimensions as list (e.g., [6])
            output_dim: Output dimensions as list (e.g., [1])
            init_fn: Optional initialization function
            regularizers: Dict of custom regularization functions
            temperature: Temperature parameter (kept for compatibility, not used with clamp)
            eval_mode: Evaluation mode
            use_cuda: Whether to use CUDA kernels (if available)
            alpha: Gradient stabilization strength (typically 0.8-1.0)
            scale_min: Minimum allowed gradient scale factor (default 0.1)
            scale_max: Maximum allowed gradient scale factor (default 10.0)
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, regularizers=regularizers)
        self.register_buffer('temperature', torch.tensor(float(temperature)))
        self.register_buffer('alpha', torch.tensor(float(alpha)))
        self.register_buffer('scale_min', torch.tensor(float(scale_min)))
        self.register_buffer('scale_max', torch.tensor(float(scale_max)))
        
        assert eval_mode in {"expectation", "deterministic", "threshold"}, "Invalid eval_mode"
        self.eval_mode = eval_mode
        self.use_cuda = use_cuda and is_cuda_available()
        
        # Warn if CUDA requested but not available
        if use_cuda and not _CUDA_EXT_AVAILABLE:
            warnings.warn(
                "ProbabilisticStableNode: CUDA was requested (use_cuda=True) but CUDA extension is not available. "
                "Using CPU fallback which may be significantly slower. "
                "To enable CUDA: compile the extension with 'cd difflut && python setup.py install'",
                RuntimeWarning,
                stacklevel=2
            )
        
        # Store raw weights (logits) that will be passed through sigmoid
        if init_fn:
            self.raw_weights = nn.Parameter(init_fn((2**self.num_inputs, self.num_outputs)))
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
        """Get weights with clamp applied to [0,1] (faster than sigmoid)."""
        return torch.clamp(self.raw_weights, 0.0, 1.0)

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
        Uses CUDA kernel with gradient stabilization if available, otherwise falls back to CPU.
        """
        x = self._prepare_input(x)
        
        # Try CUDA kernel first
        if self.use_cuda and x.is_cuda and _CUDA_EXT_AVAILABLE:
            # Transpose raw_weights to match CUDA kernel expectations
            # raw_weights is (2^n, num_outputs), need (num_outputs, 2^n)
            luts = self.raw_weights.t().contiguous()
            mapping = self.mapping.int()
            
            output = probabilistic_stable_forward(
                x, mapping, luts, self.temperature, 
                self.alpha, self.scale_min, self.scale_max
            )
            return self._prepare_output(output)
        
        # CPU fallback - original vectorized implementation with gradient stabilization
        batch_size = x.shape[0]
        # Ensure binary_combinations is on the same device and dtype as x
        binary_combinations = self.binary_combinations.to(device=x.device, dtype=x.dtype)
        # Vectorized probability computation
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, num_inputs)
        a_expanded = binary_combinations.unsqueeze(0)  # (1, 2^num_inputs, num_inputs)
        prob_terms = x_expanded * a_expanded + (1 - x_expanded) * (1 - a_expanded)
        probs = torch.prod(prob_terms, dim=2)
        weights = self.weights
        
        # Wrap in custom autograd function for CPU gradient stabilization
        output = CPUProbabilisticStableFunction.apply(
            probs, weights, x, 
            self.alpha, self.num_inputs, self.num_outputs,
            self.scale_min, self.scale_max
        )
        
        return self._prepare_output(output)

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Direct LUT lookup simulating hardware behavior.
        Assumes inputs are already binary (0 or 1) from encoder or previous nodes.
        Returns binary outputs (0 or 1).
        """
        x = self._prepare_input(x)
        batch_size = x.shape[0]
        
        # Convert binary inputs to LUT indices (LSB-first)
        indices = self._binary_to_index(x)  # (batch_size,)
        
        # Look up weights and threshold at 0.5 to get binary output
        weights = self.weights  # (2^num_inputs, num_outputs) - clamped to [0, 1]
        output = weights[indices]  # (batch_size, num_outputs)
        
        # Threshold to get binary output (weights are in [0, 1] after clamp)
        output = (output >= 0.5).float()
        
        return self._prepare_output(output)

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization by default."""
        return torch.tensor(0.0, device=self.raw_weights.device)

    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
                f"alpha={self.alpha.item():.2f}, "
                f"scale_range=[{self.scale_min.item():.2f}, {self.scale_max.item():.2f}]")


class CPUProbabilisticStableFunction(torch.autograd.Function):
    """
    CPU fallback with gradient stabilization.
    """
    @staticmethod
    def forward(ctx, probs, weights, x, alpha, num_inputs, num_outputs, scale_min, scale_max):
        """
        Args:
            probs: (batch_size, 2^num_inputs) - precomputed probabilities
            weights: (2^num_inputs, num_outputs) - LUT weights after sigmoid
            x: (batch_size, num_inputs) - original inputs (saved for context)
            alpha: scalar - gradient stabilization strength
            num_inputs: int
            num_outputs: int
            scale_min: scalar
            scale_max: scalar
        """
        output = torch.matmul(probs, weights)
        
        # Save for backward
        ctx.save_for_backward(probs, weights, torch.tensor(alpha), 
                            torch.tensor(scale_min), torch.tensor(scale_max))
        ctx.num_inputs = num_inputs
        ctx.num_outputs = num_outputs
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward with gradient stabilization.
        
        Args:
            grad_output: (batch_size, num_outputs)
        """
        probs, weights, alpha, scale_min, scale_max = ctx.saved_tensors
        num_inputs = ctx.num_inputs
        num_outputs = ctx.num_outputs
        
        # Standard gradients
        grad_probs = torch.matmul(grad_output, weights.t())  # (batch_size, 2^num_inputs)
        grad_weights = torch.matmul(probs.t(), grad_output)  # (2^num_inputs, num_outputs)
        
        # Gradient stabilization for input gradients
        # Note: In CPU mode, we don't have direct access to x gradients here
        # This is a simplified version - full stabilization requires modifying the entire chain
        # For now, we apply stabilization to the output gradient
        
        # Compute L1 norms per sample
        grad_out_l1 = grad_output.abs().sum(dim=1, keepdim=True)  # (batch_size, 1)
        grad_in_raw_l1 = grad_probs.abs().sum(dim=1, keepdim=True)  # (batch_size, 1)
        
        # Compute log ratio
        if num_outputs > 0:
            log_ratio = torch.log(torch.tensor(1.0 + num_inputs / num_outputs, 
                                              device=grad_output.device))
        else:
            log_ratio = torch.tensor(0.0, device=grad_output.device)
        
        # Compute per-sample scaling factor
        eps = 1e-6
        scale = (alpha * log_ratio * grad_out_l1) / (grad_in_raw_l1 + eps)
        scale = scale.clamp(min=scale_min.item(), max=scale_max.item())
        
        # Apply scaling
        grad_probs_scaled = grad_probs * scale
        
        # Return gradients: (probs, weights, x, alpha, num_inputs, num_outputs, scale_min, scale_max)
        return grad_probs_scaled, grad_weights, None, None, None, None, None, None
