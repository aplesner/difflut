import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable
import warnings
from .base_node import BaseNode
from ..registry import register_node
from .cuda import is_cuda_available

# Try to import the fourier CUDA extension
try:
    import fourier_cuda as _fourier_cuda_module
    _FOURIER_CUDA_EXT_AVAILABLE = True
except ImportError:
    _FOURIER_CUDA_EXT_AVAILABLE = False
    _fourier_cuda_module = None
    warnings.warn(
        "CUDA extension 'fourier_cuda' not available. FourierNode will use slower CPU fallback. "
        "For better performance, compile the CUDA extension using: "
        "'cd difflut && python setup.py install'. "
        "To suppress this warning: warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut.nodes.fourier_node')",
        RuntimeWarning,
        stacklevel=2
    )


class FourierFunction(torch.autograd.Function):
    """
    PyTorch autograd function wrapper for Fourier CUDA kernels.
    """
    @staticmethod
    def forward(ctx, input, frequencies, amplitudes, phases, bias, max_amplitude, use_eval):
        """
        Forward pass using CUDA kernel with 3D tensors.
        
        Args:
            input: (batch_size, layer_size, num_inputs) float tensor
            frequencies: (num_frequencies, num_inputs) float tensor
            amplitudes: (layer_size, num_frequencies, output_dim) float tensor
            phases: (layer_size, num_frequencies, output_dim) float tensor
            bias: (layer_size, output_dim) float tensor
            max_amplitude: float - maximum amplitude parameter
            use_eval: bool - whether to use Heaviside function (eval mode)
        
        Returns:
            output: (batch_size, layer_size, output_dim) float tensor
        """
        if not _FOURIER_CUDA_EXT_AVAILABLE:
            raise RuntimeError("Fourier CUDA extension not available. Please compile fourier_cuda extension.")
        
        # Ensure correct dtypes and contiguity
        input = input.contiguous().float()
        frequencies = frequencies.contiguous().float()
        amplitudes = amplitudes.contiguous().float()
        phases = phases.contiguous().float()
        bias = bias.contiguous().float()
        
        # Call appropriate CUDA forward kernel (now handles 3D tensors natively)
        if use_eval:
            output = _fourier_cuda_module.forward_eval(input, frequencies, amplitudes, phases, bias, max_amplitude)
        else:
            output = _fourier_cuda_module.forward(input, frequencies, amplitudes, phases, bias, max_amplitude)
        
        # Save for backward
        ctx.save_for_backward(input, frequencies, amplitudes, phases)
        ctx.max_amplitude = max_amplitude
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using CUDA kernel with 3D tensors.
        
        Args:
            grad_output: (batch_size, layer_size, output_dim) gradient tensor
        
        Returns:
            Gradients for (input, frequencies, amplitudes, phases, bias, max_amplitude, use_eval)
        """
        if not _FOURIER_CUDA_EXT_AVAILABLE:
            raise RuntimeError("Fourier CUDA extension not available. Please compile fourier_cuda extension.")
        
        input, frequencies, amplitudes, phases = ctx.saved_tensors
        max_amplitude = ctx.max_amplitude
        
        # Ensure contiguity
        grad_output = grad_output.contiguous().float()
        
        # Call CUDA backward kernel (now handles 3D tensors natively)
        grad_input, grad_amplitudes, grad_phases, grad_bias = _fourier_cuda_module.backward(
            input, frequencies, amplitudes, phases, grad_output, max_amplitude
        )
        
        # Return gradients (None for frequencies and max_amplitude as they don't need gradients)
        return grad_input, None, grad_amplitudes, grad_phases, grad_bias, None, None


def fourier_forward(input, frequencies, amplitudes, phases, bias, max_amplitude, use_eval=False):
    """
    Fourier forward pass with automatic differentiation support.
    
    Args:
        input: (batch_size, layer_size, num_inputs) tensor
        frequencies: (num_frequencies, num_inputs) tensor
        amplitudes: (layer_size, num_frequencies, output_dim) tensor
        phases: (layer_size, num_frequencies, output_dim) tensor
        bias: (layer_size, output_dim) tensor
        max_amplitude: float - maximum amplitude parameter
        use_eval: bool - whether to use Heaviside function for evaluation
    
    Returns:
        output: (batch_size, layer_size, output_dim) tensor
    """
    if _FOURIER_CUDA_EXT_AVAILABLE and input.is_cuda:
        return FourierFunction.apply(input, frequencies, amplitudes, phases, bias, max_amplitude, use_eval)
    else:
        # CPU fallback - Fourier computation with per-layer-node parameters
        batch_size, layer_size, input_dim = input.shape
        output_dim = bias.shape[1]
        num_frequencies = frequencies.shape[0]
        
        # Input is assumed to already be in [0, 1] range
        # Apply heaviside for eval mode only
        if use_eval:
            x_processed = (input > 0.5).float()
        else:
            x_processed = input  # No sigmoid - input already in [0,1]
        
        # Compute <k, x> for all frequencies
        # x_processed: (batch_size, layer_size, num_inputs)
        # frequencies: (num_freq, num_inputs)
        # Result: (batch_size, layer_size, num_freq)
        dot_products = torch.einsum('bli,fi->blf', x_processed, frequencies)
        
        # Compute angles: 2π * <k, x>
        angles = 2 * np.pi * dot_products  # (batch_size, layer_size, num_freq)
        
        # Normalize amplitudes to ensure bounded output
        # amplitudes: (layer_size, num_frequencies, output_dim)
        # Σ|w_k| ≤ max_amplitude per layer node
        normalized_amplitudes = amplitudes / (amplitudes.sum(dim=1, keepdim=True) + 1e-8) * max_amplitude
        
        # Compute: Σ_k |w_k| * cos(2π * <k, x> + φ_k)
        # angles: (batch_size, layer_size, num_freq)
        # phases: (layer_size, num_freq, output_dim)
        # normalized_amplitudes: (layer_size, num_freq, output_dim)
        
        # Add phases: (batch_size, layer_size, num_freq, output_dim)
        phase_shifted_angles = angles.unsqueeze(-1) + phases.unsqueeze(0)
        
        # Compute cosines
        cosines = torch.cos(phase_shifted_angles)  # (batch_size, layer_size, num_freq, output_dim)
        
        # Weighted sum over frequencies
        # cosines: (batch_size, layer_size, num_freq, output_dim)
        # normalized_amplitudes: (layer_size, num_freq, output_dim)
        output = torch.einsum('blfo,lfo->blo', cosines, normalized_amplitudes)
        
        # Add bias: (layer_size, output_dim)
        output = output + bias.unsqueeze(0)
        
        # Clamp to [0, 1]
        output = torch.clamp(output, 0.0, 1.0)
        
        return output


@register_node("fourier")
class FourierNode(BaseNode):
    """
    Fourier LUT Node implementing a bounded discrete Fourier transform.
    
    The output is computed as:
        y = 0.5 + Σ_k Re(w_k * exp(i * 2π * <k, x>))
    
    Where:
    - k are frequency vectors on the hypercube corners {0,1}^n
    - w_k are complex weights with Hermitian symmetry (w_{-k} = conj(w_k))
    - x is the input vector
    - The output is guaranteed to be real and bounded in [0, 1]
    
    The Hermitian symmetry ensures the imaginary parts cancel, giving a real output.
    The weights are normalized to keep the amplitude bounded.
    
    Now supports per-layer-node parameters for better memory access patterns.
    """
    
    def __init__(self, 
                 input_dim: int | None = None,
                 output_dim: int | None = None,
                 layer_size: int | None = None,
                 use_all_frequencies: bool = True,
                 max_amplitude: float = 0.5,
                 use_cuda: bool = True,
                 regularizers: dict | None = None,
                 init_fn: Optional[Callable] = None,
                 init_kwargs: dict | None = None):
        """
        Args:
            input_dim: Input dimensions (e.g., 6)
            output_dim: Output dimensions (e.g., 1)
            layer_size: Number of parallel nodes in the layer (e.g., 128)
            use_all_frequencies: If True, use all 2^n frequency vectors.
                                If False, use only a subset for efficiency.
            max_amplitude: Maximum amplitude of oscillation (default 0.5 for [0,1] output)
            use_cuda: Whether to use CUDA kernels (if available)
            regularizers: Dict of custom regularization functions
            init_fn: Optional initialization function. Should take (param: torch.Tensor, **kwargs)
            init_kwargs: Keyword arguments for init_fn
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, layer_size=layer_size,
                         regularizers=regularizers, init_fn=init_fn, init_kwargs=init_kwargs)
        self.use_all_frequencies = use_all_frequencies
        self.max_amplitude = max_amplitude
        self.use_cuda = use_cuda and is_cuda_available()
        
        # Generate frequency vectors k
        # CRITICAL: We cannot use integer frequencies {0,1}^n because at binary corners,
        # all dot products are integers, giving angles that are multiples of 2π.
        # Since cos(2πn) = 1 for all n, this makes all Fourier terms identical at corners!
        # Solution: Use fractional frequencies (scaled by 0.5) so we get proper variation.
        self.num_frequencies = 2 ** self.num_inputs if use_all_frequencies else min(2 ** self.num_inputs, 32)
        
        # Create frequency vectors (shared across all layer_size nodes)
        if use_all_frequencies:
            # Use 0.5-scaled corners to avoid integer dot products at binary inputs
            # This gives angles in {0, π, 2π, ...} instead of {0, 2π, 4π, ...}
            frequencies = []
            for i in range(2 ** self.num_inputs):
                k = [0.5 * ((i >> j) & 1) for j in reversed(range(self.num_inputs))]  # Scale by 0.5
                frequencies.append(k)
            frequencies = torch.tensor(frequencies, dtype=torch.float32)
        else:
            # Sample a subset of frequencies (also scaled by 0.5)
            frequencies = 0.5 * torch.randint(0, 2, (self.num_frequencies, self.num_inputs), dtype=torch.float32)
        
        self.register_buffer('frequencies', frequencies)  # Shape: (num_freq, num_inputs)
        
        # Initialize complex weights as separate real and imaginary parts
        # Now with per-layer-node parameters
        # Shape: (layer_size, num_frequencies, output_dim)
        
        # Amplitudes (always positive) - create with defaults, then apply init_fn if provided
        self.amplitudes = nn.Parameter(torch.rand(self.layer_size, self.num_frequencies, self.num_outputs) * 0.1)
        self._apply_init_fn(self.amplitudes, name="amplitudes")
        
        # Phases (in radians)
        self.phases = nn.Parameter(
            torch.rand(self.layer_size, self.num_frequencies, self.num_outputs) * 2 * np.pi - np.pi
        )
        self._apply_init_fn(self.phases, name="phases")
        
        # Bias term (initialized at 0.5 to center output in [0, 1])
        # Shape: (layer_size, output_dim)
        self.bias = nn.Parameter(torch.full((self.layer_size, self.num_outputs), 0.5))
        self._apply_init_fn(self.bias, name="bias")
    
    def _compute_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Fourier sum with per-layer-node parameters.
        
        Args:
            x: Input tensor (batch_size, layer_size, num_inputs) in [0, 1]
        Returns:
            Output tensor (batch_size, layer_size, output_dim)
        """
        batch_size, layer_size, input_dim = x.shape
        
        # Verify layer_size matches
        if layer_size != self.layer_size:
            raise ValueError(
                f"Input layer_size {layer_size} does not match node's layer_size {self.layer_size}"
            )
        
        # Input is assumed to already be in [0, 1] range
        
        # Compute <k, x> for all frequencies
        # x: (batch_size, layer_size, num_inputs)
        # frequencies: (num_freq, num_inputs)
        # Result: (batch_size, layer_size, num_freq)
        dot_products = torch.einsum('bli,fi->blf', x, self.frequencies)
        
        # Compute angles: 2π * <k, x>
        angles = 2 * np.pi * dot_products  # (batch_size, layer_size, num_freq)
        
        # Normalize amplitudes to ensure bounded output
        # amplitudes: (layer_size, num_frequencies, output_dim)
        # Σ|w_k| ≤ max_amplitude per layer node
        normalized_amplitudes = self.amplitudes / (self.amplitudes.sum(dim=1, keepdim=True) + 1e-8) * self.max_amplitude
        
        # Compute: Σ_k |w_k| * cos(2π * <k, x> + φ_k)
        # angles: (batch_size, layer_size, num_freq)
        # phases: (layer_size, num_freq, output_dim)
        # normalized_amplitudes: (layer_size, num_freq, output_dim)
        
        # Add phases: (batch_size, layer_size, num_freq, output_dim)
        phase_shifted_angles = angles.unsqueeze(-1) + self.phases.unsqueeze(0)
        
        # Compute cosines
        cosines = torch.cos(phase_shifted_angles)  # (batch_size, layer_size, num_freq, output_dim)
        
        # Weighted sum over frequencies
        # cosines: (batch_size, layer_size, num_freq, output_dim)
        # normalized_amplitudes: (layer_size, num_freq, output_dim)
        output = torch.einsum('blfo,lfo->blo', cosines, normalized_amplitudes)
        
        # Add bias: (layer_size, output_dim)
        output = output + self.bias.unsqueeze(0)
        
        # Clamp to [0, 1]
        output = torch.clamp(output, 0.0, 1.0)
        
        return output
    
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training.
        Uses CUDA kernels if available, otherwise falls back to Python implementation.
        
        Args:
            x: Input tensor (batch_size, layer_size, num_inputs)
        Returns:
            Output tensor (batch_size, layer_size, output_dim)
        """
        batch_size, layer_size, input_dim = x.shape
        
        # Verify layer_size matches
        if layer_size != self.layer_size:
            raise ValueError(
                f"Input layer_size {layer_size} does not match node's layer_size {self.layer_size}"
            )
        
        # Use CUDA-accelerated forward if available, otherwise Python fallback
        if self.use_cuda and x.is_cuda and _FOURIER_CUDA_EXT_AVAILABLE:
            output = fourier_forward(
                x, self.frequencies, self.amplitudes, self.phases, self.bias,
                self.max_amplitude, use_eval=False
            )
        else:
            # Python fallback (already implemented in _compute_output)
            output = self._compute_output(x)
        
        return output
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Uses Heaviside thresholding for binary behavior.
        Uses CUDA kernels if available with Heaviside in the kernel.
        
        Args:
            x: Input tensor (batch_size, layer_size, num_inputs)
        Returns:
            Output tensor (batch_size, layer_size, output_dim)
        """
        batch_size, layer_size, input_dim = x.shape
        
        # Verify layer_size matches
        if layer_size != self.layer_size:
            raise ValueError(
                f"Input layer_size {layer_size} does not match node's layer_size {self.layer_size}"
            )
        
        # Use CUDA-accelerated forward_eval if available
        if self.use_cuda and x.is_cuda and _FOURIER_CUDA_EXT_AVAILABLE:
            output = fourier_forward(
                x, self.frequencies, self.amplitudes, self.phases, self.bias,
                self.max_amplitude, use_eval=True
            )
        else:
            # Python fallback: compute output then discretize
            output = self._compute_output(x)
            # Discretize: Heaviside at 0.5 since output is in [0,1]
            output = (output >= 0.5).float()
        
        return output
    
    def _builtin_regularization(self) -> torch.Tensor:
        """
        Regularization to encourage sparse frequency usage.
        L1 penalty on amplitudes to prefer simpler functions.
        """
        return 0.01 * torch.sum(torch.abs(self.amplitudes))
    
    def get_dominant_frequencies(self, top_k: int = 5) -> torch.Tensor:
        """
        Get the top-k most important frequencies (highest amplitudes).
        
        Args:
            top_k: Number of top frequencies to return
        Returns:
            Tensor of shape (top_k, num_inputs) with the dominant frequency vectors
        """
        with torch.no_grad():
            # Average amplitude across output dimensions
            avg_amplitudes = self.amplitudes.mean(dim=1)
            top_indices = torch.topk(avg_amplitudes, min(top_k, self.num_frequencies)).indices
            return self.frequencies[top_indices]
    
    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
                f"num_frequencies={self.num_frequencies}, max_amplitude={self.max_amplitude}")

