import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from .base_node import BaseNode
from ..registry import register_node
from .cuda import is_cuda_available, fourier_forward


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
    """
    
    def __init__(self, 
                 num_inputs: int, 
                 output_dim: int = 1,
                 use_all_frequencies: bool = True,
                 max_amplitude: float = 0.5,
                 use_cuda: bool = True,
                 regularizers: dict = None):
        """
        Args:
            num_inputs: Number of inputs to the node (n)
            output_dim: Number of output dimensions
            use_all_frequencies: If True, use all 2^n frequency vectors.
                                If False, use only a subset for efficiency.
            max_amplitude: Maximum amplitude of oscillation (default 0.5 for [0,1] output)
            use_cuda: Whether to use CUDA kernels (if available)
            regularizers: Dict of custom regularization functions
        """
        super().__init__(num_inputs=num_inputs, regularizers=regularizers)
        self.output_dim = output_dim
        self.use_all_frequencies = use_all_frequencies
        self.max_amplitude = max_amplitude
        self.use_cuda = use_cuda and is_cuda_available()
        
        # Generate frequency vectors k (corners of hypercube {0,1}^n)
        # These are the same as binary combinations in the LUT
        self.num_frequencies = 2 ** num_inputs if use_all_frequencies else min(2 ** num_inputs, 32)
        
        # Create frequency vectors
        if use_all_frequencies:
            # All 2^n corners of the hypercube
            frequencies = []
            for i in range(2 ** num_inputs):
                k = []
                for j in range(num_inputs):
                    k.append((i >> j) & 1)
                frequencies.append(k)
            frequencies = torch.tensor(frequencies, dtype=torch.float32)
        else:
            # Sample a subset of frequencies
            frequencies = torch.randint(0, 2, (self.num_frequencies, num_inputs), dtype=torch.float32)
        
        self.register_buffer('frequencies', frequencies)  # Shape: (num_freq, num_inputs)
        
        # Initialize complex weights as separate real and imaginary parts
        # For Hermitian symmetry, we only store weights for k and compute -k automatically
        # However, since our frequencies are from {0,1}^n, we use a different approach:
        # We store amplitude and phase for each frequency
        
        # Amplitudes (always positive)
        self.amplitudes = nn.Parameter(
            torch.rand(self.num_frequencies, output_dim) * 0.1
        )
        
        # Phases (in radians)
        self.phases = nn.Parameter(
            torch.rand(self.num_frequencies, output_dim) * 2 * np.pi - np.pi
        )
        
        # Bias term (initialized at 0.5 to center output in [0, 1])
        self.bias = nn.Parameter(torch.full((output_dim,), 0.5))
    
    def _compute_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Fourier sum.
        
        Args:
            x: Input tensor (batch_size, num_inputs) in [0, 1]
        Returns:
            Output tensor (batch_size, output_dim) in [0, 1]
        """
        batch_size = x.shape[0]
        
        # Ensure input is in [0, 1] range
        x = torch.sigmoid(x)  # Map to [0, 1]
        
        # Compute <k, x> for all frequencies
        # x: (batch_size, num_inputs)
        # frequencies: (num_freq, num_inputs)
        # Result: (batch_size, num_freq)
        dot_products = torch.matmul(x, self.frequencies.t())  # (batch_size, num_freq)
        
        # Compute exp(i * 2π * <k, x>) = cos(2π<k,x>) + i*sin(2π<k,x>)
        # We only need the real part for our final output
        angles = 2 * np.pi * dot_products  # (batch_size, num_freq)
        
        # Normalize amplitudes to ensure bounded output
        # Σ|w_k| ≤ max_amplitude
        normalized_amplitudes = self.amplitudes / (self.amplitudes.sum(dim=0, keepdim=True) + 1e-8) * self.max_amplitude
        
        # Compute: Σ_k |w_k| * cos(2π * <k, x> + φ_k)
        # This is equivalent to the real part of: Σ_k w_k * exp(i * 2π * <k, x>)
        # where w_k = |w_k| * exp(i * φ_k)
        
        # For each output dimension
        outputs = []
        for dim in range(self.output_dim):
            # angles: (batch_size, num_freq)
            # phases: (num_freq, output_dim)
            # normalized_amplitudes: (num_freq, output_dim)
            
            phase_shifted_angles = angles + self.phases[:, dim].unsqueeze(0)  # (batch_size, num_freq)
            cosines = torch.cos(phase_shifted_angles)  # (batch_size, num_freq)
            
            # Weighted sum
            weighted_sum = torch.matmul(cosines, normalized_amplitudes[:, dim])  # (batch_size,)
            outputs.append(weighted_sum)
        
        output = torch.stack(outputs, dim=1)  # (batch_size, output_dim)
        
        # Add bias and ensure output is in [0, 1]
        output = output + self.bias
        output = torch.clamp(output, 0.0, 1.0)
        
        return output
    
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training.
        Uses CUDA kernel if available, otherwise falls back to Python.
        
        Args:
            x: Input tensor (batch_size, num_inputs) or (batch_size, 1, num_inputs)
        Returns:
            Output tensor (batch_size, output_dim) or (batch_size,) if output_dim=1
        """
        # Handle different input dimensions
        if x.dim() == 3:
            x = x.squeeze(1)  # Remove middle dimension
        
        # Use CUDA kernels if available and on GPU
        if self.use_cuda and x.is_cuda:
            # Ensure input is contiguous and float32
            x_cont = x.contiguous().float()
            
            # Call fourier CUDA kernel
            output = fourier_forward(
                x_cont,
                self.frequencies,
                self.amplitudes,
                self.phases,
                self.bias,
                self.max_amplitude,
                use_eval=False
            )
        else:
            # Fallback to Python implementation
            output = self._compute_output(x)
        
        # Squeeze if output_dim is 1
        if self.output_dim == 1:
            output = output.squeeze(-1)
        
        return output
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during evaluation using Heaviside function.
        Uses base_node implementation (Heaviside thresholding).
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # Use CUDA with Heaviside if available
        if self.use_cuda and x.is_cuda:
            x_cont = x.contiguous().float()
            
            # Call fourier CUDA kernel with eval mode (Heaviside)
            output = fourier_forward(
                x_cont,
                self.frequencies,
                self.amplitudes,
                self.phases,
                self.bias,
                self.max_amplitude,
                use_eval=True
            )
        else:
            # Fallback: Use base_node implementation with Heaviside
            # Apply Heaviside to input (binary threshold at 0.5)
            x_heaviside = (x > 0.5).float()
            output = self._compute_output(x_heaviside)
        
        if self.output_dim == 1:
            output = output.squeeze(-1)
        
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
        return (f"num_inputs={self.num_inputs}, output_dim={self.output_dim}, "
                f"num_frequencies={self.num_frequencies}, max_amplitude={self.max_amplitude}")


@register_node("fourier_hermitian")
class FourierHermitianNode(BaseNode):
    """
    Fourier LUT Node with explicit Hermitian symmetry enforcement.
    
    This version explicitly enforces w_{-k} = conj(w_k) by only storing
    weights for half the frequencies and computing the conjugates.
    
    This is more mathematically rigorous but computationally more complex.
    """
    
    def __init__(self, 
                 num_inputs: int, 
                 output_dim: int = 1,
                 max_amplitude: float = 0.5,
                 regularizers: dict = None):
        """
        Args:
            num_inputs: Number of inputs to the node (n)
            output_dim: Number of output dimensions
            max_amplitude: Maximum amplitude of oscillation
            regularizers: Dict of custom regularization functions
        """
        super().__init__(num_inputs=num_inputs, regularizers=regularizers)
        self.output_dim = output_dim
        self.max_amplitude = max_amplitude
        
        # Generate all frequency vectors k as integer vectors
        # Map from {0,1}^n to {-1,0,1,...} for proper frequency interpretation
        num_freq = 2 ** num_inputs
        frequencies = []
        for i in range(num_freq):
            k = []
            for j in range(num_inputs):
                bit = (i >> j) & 1
                k.append(bit)  # Keep in {0,1} for now
            frequencies.append(k)
        
        frequencies = torch.tensor(frequencies, dtype=torch.float32)
        self.register_buffer('frequencies', frequencies)
        
        # For Hermitian symmetry, we only need to store weights for half + DC
        # The DC component (k=0) must be real
        # For k ≠ 0, we store w_k and enforce w_{-k} = conj(w_k)
        
        # Store complex weights as real and imaginary parts
        # Only store unique frequencies (half of them + DC)
        half_freq = (num_freq + 1) // 2
        
        # Real parts (all frequencies contribute)
        self.weights_real = nn.Parameter(torch.randn(half_freq, output_dim) * 0.1)
        
        # Imaginary parts (DC component should be zero)
        self.weights_imag = nn.Parameter(torch.randn(half_freq, output_dim) * 0.1)
        
        # Bias
        self.bias = nn.Parameter(torch.full((output_dim,), 0.5))
    
    def _compute_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Fourier sum with Hermitian symmetry.
        
        Args:
            x: Input tensor (batch_size, num_inputs)
        Returns:
            Output tensor (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        x = torch.sigmoid(x)
        
        # Compute all dot products
        dot_products = torch.matmul(x, self.frequencies.t())
        angles = 2 * np.pi * dot_products
        
        # For simplicity, compute the full sum using symmetry
        # DC component (k=0, must be real)
        output = self.weights_real[0:1, :].expand(batch_size, -1)
        
        # Other frequencies: sum both k and -k contributions
        # For each k in our half, add: 2 * Re(w_k * exp(i*2π*<k,x>))
        for k_idx in range(1, self.weights_real.shape[0]):
            w_real = self.weights_real[k_idx]
            w_imag = self.weights_imag[k_idx]
            
            # exp(i*θ) = cos(θ) + i*sin(θ)
            cos_vals = torch.cos(angles[:, k_idx])
            sin_vals = torch.sin(angles[:, k_idx])
            
            # w * exp(i*θ) = (w_r + i*w_i)(cos + i*sin) = (w_r*cos - w_i*sin) + i*(w_r*sin + w_i*cos)
            # Real part: w_r*cos - w_i*sin
            real_part = w_real * cos_vals.unsqueeze(1) - w_imag * sin_vals.unsqueeze(1)
            
            # Add 2 * real part (accounts for both k and -k)
            output = output + 2 * real_part
        
        # Normalize to keep bounded
        # Scale by max_amplitude / expected_max
        scale = self.max_amplitude / (torch.sum(torch.abs(self.weights_real)) + torch.sum(torch.abs(self.weights_imag)) + 1e-8)
        output = output * scale.unsqueeze(0)
        
        # Add bias and clamp
        output = output + self.bias
        output = torch.clamp(output, 0.0, 1.0)
        
        return output
    
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass during training."""
        if x.dim() == 3:
            x = x.squeeze(1)
        
        output = self._compute_output(x)
        
        if self.output_dim == 1:
            output = output.squeeze(-1)
        
        return output
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass during evaluation."""
        return self.forward_train(x)
    
    def _builtin_regularization(self) -> torch.Tensor:
        """L1 regularization on weights."""
        return 0.01 * (torch.sum(torch.abs(self.weights_real)) + torch.sum(torch.abs(self.weights_imag)))
    
    def extra_repr(self) -> str:
        return f"num_inputs={self.num_inputs}, output_dim={self.output_dim}, max_amplitude={self.max_amplitude}"


# Export
__all__ = ['FourierNode', 'FourierHermitianNode']
