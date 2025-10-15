"""
CUDA kernels for DiffLUT nodes.
"""
import torch

def is_cuda_available():
    """Check if CUDA is available for use."""
    return torch.cuda.is_available()

# Try to import the compiled CUDA extension
try:
    import efd_cuda as _efd_cuda_module
    _CUDA_EXT_AVAILABLE = True
except ImportError as e:
    _CUDA_EXT_AVAILABLE = False
    _efd_cuda_module = None
    import warnings
    warnings.warn(f"CUDA extension efd_cuda not available: {e}. Using CPU fallback.")

# Try to import the hybrid CUDA extension
try:
    import hybrid_cuda as _hybrid_cuda_module
    _HYBRID_CUDA_EXT_AVAILABLE = True
except ImportError as e:
    _HYBRID_CUDA_EXT_AVAILABLE = False
    _hybrid_cuda_module = None
    import warnings
    warnings.warn(f"CUDA extension hybrid_cuda not available: {e}. Using CPU fallback.")

# Try to import the fourier CUDA extension
try:
    import fourier_cuda as _fourier_cuda_module
    _FOURIER_CUDA_EXT_AVAILABLE = True
except ImportError as e:
    _FOURIER_CUDA_EXT_AVAILABLE = False
    _fourier_cuda_module = None
    import warnings
    warnings.warn(f"CUDA extension fourier_cuda not available: {e}. Using CPU fallback.")


class EFDFunction(torch.autograd.Function):
    """
    PyTorch autograd function wrapper for EFD CUDA kernels.
    """
    @staticmethod
    def forward(ctx, input, mapping, luts, alpha, beta):
        """
        Forward pass using CUDA kernel.
        
        Args:
            input: (batch_size, input_length) float tensor
            mapping: (num_luts, n) int tensor - mapping of inputs to each LUT
            luts: (num_luts, 2^n) float tensor - LUT values
            alpha: EFD parameter for finite difference
            beta: EFD parameter for distance decay
        
        Returns:
            output: (batch_size, num_luts) float tensor
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Please compile efd_cuda extension.")
        
        # Ensure correct dtypes and contiguity
        input = input.contiguous().float()
        mapping = mapping.contiguous().int()
        luts = luts.contiguous().float()
        
        # Call CUDA forward kernel
        output = _efd_cuda_module.forward(input, mapping, luts)
        
        # Save for backward
        ctx.save_for_backward(input, mapping, luts)
        ctx.alpha = alpha
        ctx.beta = beta
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using CUDA kernel.
        
        Args:
            grad_output: (batch_size, num_luts) gradient tensor
        
        Returns:
            Gradients for (input, mapping, luts, alpha, beta)
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Please compile efd_cuda extension.")
        
        input, mapping, luts = ctx.saved_tensors
        alpha = ctx.alpha
        beta = ctx.beta
        
        # Ensure contiguity
        grad_output = grad_output.contiguous().float()
        
        # Call CUDA backward kernel
        grad_input, grad_luts = _efd_cuda_module.backward(
            input, mapping, luts, alpha, beta, grad_output
        )
        
        # Return gradients (None for mapping, alpha, beta as they don't need gradients)
        return grad_input, None, grad_luts, None, None


def efd_forward(input, mapping, luts, alpha, beta):
    """
    EFD forward pass with automatic differentiation support.
    
    Args:
        input: (batch_size, input_length) tensor
        mapping: (num_luts, n) int tensor
        luts: (num_luts, 2^n) tensor
        alpha: EFD parameter
        beta: EFD parameter
    
    Returns:
        output: (batch_size, num_luts) tensor
    """
    if _CUDA_EXT_AVAILABLE and input.is_cuda:
        return EFDFunction.apply(input, mapping, luts, alpha, beta)
    else:
        # CPU fallback - just do a simple forward without custom backward
        # Binary threshold
        x_binary = (input > 0).float()
        batch_size = x_binary.shape[0]
        num_luts = luts.shape[0]
        n = mapping.shape[1]
        
        output = torch.zeros(batch_size, num_luts, device=input.device, dtype=input.dtype)
        
        for i in range(batch_size):
            for j in range(num_luts):
                # Compute address
                addr = 0
                for l in range(n):
                    if x_binary[i, mapping[j, l]] > 0:
                        addr |= (1 << l)
                output[i, j] = luts[j, addr]
        
        return output


class HybridFunction(torch.autograd.Function):
    """
    PyTorch autograd function wrapper for Hybrid CUDA kernels.
    Forward: Binary thresholding (DWN-style)
    Backward: Probabilistic gradients (UnboundProbabilistic-style)
    """
    @staticmethod
    def forward(ctx, input, mapping, luts, binary_combinations):
        """
        Forward pass using CUDA kernel.
        
        Args:
            input: (batch_size, input_length) float tensor
            mapping: (num_luts, n) int tensor - mapping of inputs to each LUT
            luts: (num_luts, 2^n) float tensor - LUT values
            binary_combinations: (2^n, n) float tensor - precomputed binary patterns
        
        Returns:
            output: (batch_size, num_luts) float tensor
        """
        if not _HYBRID_CUDA_EXT_AVAILABLE:
            raise RuntimeError("Hybrid CUDA extension not available. Please compile hybrid_cuda extension.")
        
        # Ensure correct dtypes and contiguity
        input = input.contiguous().float()
        mapping = mapping.contiguous().int()
        luts = luts.contiguous().float()
        binary_combinations = binary_combinations.contiguous().float()
        
        # Call CUDA forward kernel
        output = _hybrid_cuda_module.forward(input, mapping, luts)
        
        # Save for backward
        ctx.save_for_backward(input, mapping, luts, binary_combinations)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using CUDA kernel with probabilistic gradients.
        
        Args:
            grad_output: (batch_size, num_luts) gradient tensor
        
        Returns:
            Gradients for (input, mapping, luts, binary_combinations)
        """
        if not _HYBRID_CUDA_EXT_AVAILABLE:
            raise RuntimeError("Hybrid CUDA extension not available. Please compile hybrid_cuda extension.")
        
        input, mapping, luts, binary_combinations = ctx.saved_tensors
        
        # Ensure contiguity
        grad_output = grad_output.contiguous().float()
        
        # Call CUDA backward kernel
        grad_input, grad_luts = _hybrid_cuda_module.backward(
            input, mapping, luts, binary_combinations, grad_output
        )
        
        # Return gradients (None for mapping and binary_combinations as they don't need gradients)
        return grad_input, None, grad_luts, None


def hybrid_forward(input, mapping, luts, binary_combinations):
    """
    Hybrid forward pass with automatic differentiation support.
    Forward: Binary thresholding (efficient, discrete)
    Backward: Probabilistic gradients (smooth, trainable)
    
    Args:
        input: (batch_size, input_length) tensor
        mapping: (num_luts, n) int tensor
        luts: (num_luts, 2^n) tensor
        binary_combinations: (2^n, n) tensor - precomputed binary patterns
    
    Returns:
        output: (batch_size, num_luts) tensor
    """
    if _HYBRID_CUDA_EXT_AVAILABLE and input.is_cuda:
        return HybridFunction.apply(input, mapping, luts, binary_combinations)
    else:
        # CPU fallback - use Python implementation from hybrid_node
        # Binary threshold for forward
        x_binary = (input > 0).float()
        batch_size = x_binary.shape[0]
        num_luts = luts.shape[0]
        n = mapping.shape[1]
        
        output = torch.zeros(batch_size, num_luts, device=input.device, dtype=input.dtype)
        
        for i in range(batch_size):
            for j in range(num_luts):
                # Compute address
                addr = 0
                for l in range(n):
                    if x_binary[i, mapping[j, l]] > 0:
                        addr |= (1 << l)
                output[i, j] = luts[j, addr]
        
        return output


class FourierFunction(torch.autograd.Function):
    """
    PyTorch autograd function wrapper for Fourier CUDA kernels.
    """
    @staticmethod
    def forward(ctx, input, frequencies, amplitudes, phases, bias, max_amplitude, use_eval):
        """
        Forward pass using CUDA kernel.
        
        Args:
            input: (batch_size, num_inputs) float tensor
            frequencies: (num_frequencies, num_inputs) float tensor
            amplitudes: (num_frequencies, output_dim) float tensor
            phases: (num_frequencies, output_dim) float tensor
            bias: (output_dim,) float tensor
            max_amplitude: float - maximum amplitude parameter
            use_eval: bool - whether to use Heaviside function (eval mode)
        
        Returns:
            output: (batch_size, output_dim) float tensor
        """
        if not _FOURIER_CUDA_EXT_AVAILABLE:
            raise RuntimeError("Fourier CUDA extension not available. Please compile fourier_cuda extension.")
        
        # Ensure correct dtypes and contiguity
        input = input.contiguous().float()
        frequencies = frequencies.contiguous().float()
        amplitudes = amplitudes.contiguous().float()
        phases = phases.contiguous().float()
        bias = bias.contiguous().float()
        
        # Call appropriate CUDA forward kernel
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
        Backward pass using CUDA kernel.
        
        Args:
            grad_output: (batch_size, output_dim) gradient tensor
        
        Returns:
            Gradients for (input, frequencies, amplitudes, phases, bias, max_amplitude, use_eval)
        """
        if not _FOURIER_CUDA_EXT_AVAILABLE:
            raise RuntimeError("Fourier CUDA extension not available. Please compile fourier_cuda extension.")
        
        input, frequencies, amplitudes, phases = ctx.saved_tensors
        max_amplitude = ctx.max_amplitude
        
        # Ensure contiguity
        grad_output = grad_output.contiguous().float()
        
        # Call CUDA backward kernel
        grad_input, grad_amplitudes, grad_phases, grad_bias = _fourier_cuda_module.backward(
            input, frequencies, amplitudes, phases, grad_output, max_amplitude
        )
        
        # Return gradients (None for frequencies and max_amplitude as they don't need gradients)
        return grad_input, None, grad_amplitudes, grad_phases, grad_bias, None, None


def fourier_forward(input, frequencies, amplitudes, phases, bias, max_amplitude, use_eval=False):
    """
    Fourier forward pass with automatic differentiation support.
    
    Args:
        input: (batch_size, num_inputs) tensor
        frequencies: (num_frequencies, num_inputs) tensor
        amplitudes: (num_frequencies, output_dim) tensor
        phases: (num_frequencies, output_dim) tensor
        bias: (output_dim,) tensor
        max_amplitude: float - maximum amplitude parameter
        use_eval: bool - whether to use Heaviside function for evaluation
    
    Returns:
        output: (batch_size, output_dim) tensor
    """
    if _FOURIER_CUDA_EXT_AVAILABLE and input.is_cuda:
        return FourierFunction.apply(input, frequencies, amplitudes, phases, bias, max_amplitude, use_eval)
    else:
        # CPU fallback - basic Fourier computation
        import numpy as np
        
        batch_size = input.shape[0]
        output_dim = bias.shape[0]
        num_frequencies = frequencies.shape[0]
        
        # Apply sigmoid or heaviside to input
        if use_eval:
            x_processed = (input > 0.5).float()
        else:
            x_processed = torch.sigmoid(input)
        
        # Compute dot products: (batch_size, num_frequencies)
        dot_products = torch.matmul(x_processed, frequencies.t())
        
        # Compute angles: 2π * <k, x>
        angles = 2 * np.pi * dot_products  # (batch_size, num_frequencies)
        
        output = torch.zeros(batch_size, output_dim, device=input.device, dtype=input.dtype)
        
        for dim in range(output_dim):
            # Normalize amplitudes
            amplitude_sum = amplitudes[:, dim].sum()
            normalized_amplitudes = amplitudes[:, dim] / (amplitude_sum + 1e-8) * max_amplitude
            
            # Compute phase-shifted angles
            phase_shifted_angles = angles + phases[:, dim].unsqueeze(0)
            cosines = torch.cos(phase_shifted_angles)
            
            # Weighted sum
            weighted_sum = torch.matmul(cosines, normalized_amplitudes)
            output[:, dim] = weighted_sum
        
        # Add bias and clamp
        output = output + bias
        output = torch.clamp(output, 0.0, 1.0)
        
        return output


__all__ = ['is_cuda_available', 'efd_forward', 'EFDFunction', 'hybrid_forward', 'HybridFunction', 
           'fourier_forward', 'FourierFunction']
