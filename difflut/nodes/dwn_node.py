import torch
import torch.nn as nn
from typing import Optional
import warnings
from .base_node import BaseNode
from ..registry import register_node
from .cuda import is_cuda_available

# Try to import the compiled CUDA extension
try:
    import efd_cuda as _efd_cuda_module
    _CUDA_EXT_AVAILABLE = True
except ImportError:
    _CUDA_EXT_AVAILABLE = False
    _efd_cuda_module = None
    warnings.warn(
        "CUDA extension 'efd_cuda' not available. DWNNode will use slower CPU fallback. "
        "For better performance, compile the CUDA extension using: "
        "'cd difflut && python setup.py install'. "
        "To suppress this warning: warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut.nodes.dwn_node')",
        RuntimeWarning,
        stacklevel=2
    )


class EFDFunction(torch.autograd.Function):
    """
    PyTorch autograd function wrapper for EFD CUDA kernels.
    """
    @staticmethod
    def forward(ctx, input, mapping, luts):
        """
        Forward pass using CUDA kernel.
        
        Args:
            input: (batch_size, input_length) float tensor in [0, 1]
            mapping: (num_luts, n) int tensor - mapping of inputs to each LUT
            luts: (num_luts, 2^n) float tensor in [0, 1] - LUT values
        
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
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using CUDA kernel with simple finite difference.
        
        Args:
            grad_output: (batch_size, num_luts) gradient tensor
        
        Returns:
            Gradients for (input, mapping, luts)
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Please compile efd_cuda extension.")
        
        input, mapping, luts = ctx.saved_tensors
        
        # Ensure contiguity
        grad_output = grad_output.contiguous().float()
        
        # Call CUDA backward kernel
        grad_input, grad_luts = _efd_cuda_module.backward(
            input, mapping, luts, grad_output
        )
        
        # Return gradients (None for mapping as it doesn't need gradients)
        return grad_input, None, grad_luts


class EFDFunctionCPU(torch.autograd.Function):
    """
    CPU fallback for EFD with proper Extended Finite Difference backward pass.
    """
    @staticmethod
    def forward(ctx, input, mapping, luts):
        """Forward pass with binary thresholding."""
        # Binary threshold at 0.5 for [0, 1] inputs
        x_binary = (input >= 0.5).float()
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
        
        # Save for backward
        ctx.save_for_backward(input, mapping, luts)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using Extended Finite Difference (EFD)."""
        input, mapping, luts = ctx.saved_tensors
        batch_size = input.shape[0]
        input_length = input.shape[1]
        num_luts = luts.shape[0]
        n = mapping.shape[1]
        lut_size = 2 ** n
        
        grad_input = torch.zeros_like(input)
        grad_luts = torch.zeros_like(luts)
        
        for i in range(batch_size):
            for j in range(num_luts):
                # Compute current address (for LUT gradient)
                addr = 0
                for l in range(n):
                    if input[i, mapping[j, l]] >= 0.5:
                        addr |= (1 << l)
                
                # LUT gradient
                grad_luts[j, addr] += grad_output[i, j]
                
                # Input gradient using Extended Finite Difference
                for l in range(n):
                    total_gradient = 0.0
                    
                    # Create mask to exclude l-th bit
                    mask = ((1 << n) - 1) & ~(1 << l)
                    addr_masked = addr & mask
                    
                    # Iterate over all possible addresses k
                    for k in range(lut_size):
                        # Calculate Hamming distance excluding l-th bit
                        k_masked = k & mask
                        hamming_dist = bin(addr_masked ^ k_masked).count('1')
                        
                        # Get k_l (l-th bit of k)
                        k_l = (k >> l) & 1
                        
                        # Calculate sign factor: (-1)^(1-k_l)
                        sign_factor = -1.0 if k_l == 0 else 1.0
                        
                        # Get LUT value at position k
                        lut_value = luts[j, k].item()
                        
                        # Add weighted contribution
                        total_gradient += sign_factor * lut_value / (hamming_dist + 1.0)
                    
                    grad_input[i, mapping[j, l]] += total_gradient * grad_output[i, j]
        
        return grad_input, None, grad_luts


def efd_forward(input, mapping, luts):
    """
    EFD forward pass with automatic differentiation support.
    
    Args:
        input: (batch_size, input_length) tensor in [0, 1]
        mapping: (num_luts, n) int tensor
        luts: (num_luts, 2^n) tensor in [0, 1]
    
    Returns:
        output: (batch_size, num_luts) tensor
    """
    if _CUDA_EXT_AVAILABLE and input.is_cuda:
        return EFDFunction.apply(input, mapping, luts)
    else:
        # CPU fallback with proper EFD backward
        return EFDFunctionCPU.apply(input, mapping, luts)

@register_node("dwn")
class DWNNode(BaseNode):
    """
    Differentiable Weightless Neural Network node with Extended Finite Difference (EFD).
    
    Forward: Binary thresholding at 0.5 for inputs in [0, 1]
    Backward: Extended Finite Difference (EFD) - iterates over all 2^n addresses
              with Hamming distance weighting: (-1)^(1-k_j) * lut[k] / (hamming_dist + 1)
    
    Weights: Raw weights passed through sigmoid to get LUT values in [0, 1]
    """
    
    def __init__(self, 
                 input_dim: list = None,
                 output_dim: list = None,
                 use_cuda: bool = True,
                 regularizers: dict = None):
        """
        Args:
            input_dim: Input dimensions as list (e.g., [6])
            output_dim: Output dimensions as list (e.g., [1])
            use_cuda: Whether to use CUDA kernels (if available)
            regularizers: Dict of custom regularization functions
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, regularizers=regularizers)
        self.use_cuda = use_cuda and is_cuda_available()
        
        # Warn if CUDA requested but not available
        if use_cuda and not _CUDA_EXT_AVAILABLE:
            warnings.warn(
                "DWNNode: CUDA was requested (use_cuda=True) but CUDA extension is not available. "
                "Using CPU fallback which may be significantly slower. "
                "To enable CUDA: compile the extension with 'cd difflut && python setup.py install'",
                RuntimeWarning,
                stacklevel=2
            )
        
        # Initialize raw LUT weights: shape (num_outputs, 2^num_inputs)
        # Gaussian initialization around 0
        lut_size = 2 ** self.num_inputs
        self.raw_luts = nn.Parameter(
            torch.randn(self.num_outputs, lut_size) * 0.1  # Gaussian N(0, 0.1^2)
        )
        
        # Create mapping tensor (each LUT maps to all inputs in order)
        # Shape: (num_outputs, num_inputs)
        self.register_buffer('mapping', torch.arange(self.num_inputs, dtype=torch.int32).unsqueeze(0).expand(self.num_outputs, -1))
         
    def _get_luts(self) -> torch.Tensor:
        """
        Get actual LUT weights by applying sigmoid to raw weights.
        Maps from (-inf, inf) to [0, 1].
        """
        return torch.sigmoid(self.raw_luts)
    
    def _binary_to_index(self, x_binary: torch.Tensor) -> torch.Tensor:
        """
        Convert binary input to LUT index matching CUDA bit ordering.
        """
        batch_size = x_binary.shape[0]
        device = x_binary.device
        indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Build index bit by bit (LSB first)
        for l in range(self.num_inputs):
            indices = indices | (x_binary[:, l].long() << l)
        
        return indices
    
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training with automatic CUDA/CPU dispatch.
        Inputs are in [0, 1], binarized using Heaviside at 0.5.
        """
        x = self._prepare_input(x)
        
        # Get actual LUT weights via sigmoid
        luts = self._get_luts()
        
        # Use efd_forward which handles CUDA/CPU dispatch automatically
        if self.use_cuda and x.is_cuda:
            x = x.contiguous().float()
            mapping = self.mapping.int()
        else:
            mapping = self.mapping
        
        output = efd_forward(x, mapping, luts)
        return self._prepare_output(output)
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Inputs already binarized in {0, 1}.
        Output binarized to {0, 1} using Heaviside at 0.5.
        """
        x = self._prepare_input(x)
        
        # Get actual LUT weights via sigmoid
        luts = self._get_luts()
        
        # Inputs are already binarized in {0, 1}, use directly
        x_binary = x.float()
        indices = self._binary_to_index(x_binary)
        
        if self.num_outputs == 1:
            output = luts[0, indices]
        else:
            outputs = []
            for dim in range(self.num_outputs):
                outputs.append(luts[dim, indices])
            output = torch.stack(outputs, dim=1)
        
        # Binarize output: [0, 1] -> {0, 1} using Heaviside at 0.5
        # output >= 0.5 -> 1, output < 0.5 -> 0
        output = (output >= 0.5).float()
        
        return self._prepare_output(output)
    
    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization to match base CUDA implementation."""
        return torch.tensor(0.0, device=self.raw_luts.device, requires_grad=False)