import torch
import torch.nn as nn
from typing import Optional
from .base_node import BaseNode
from ..registry import register_node
from .cuda import is_cuda_available, efd_forward

@register_node("dwn")
class DWNNode(BaseNode):
    """
    Differentiable Weightless Neural Network node matching CUDA implementation.
    Uses Extended Finite Difference (EFD) for gradients.
    """
    
    def __init__(self, 
                 num_inputs: int, 
                 output_dim: int = 1,
                 alpha: Optional[float] = None,
                 beta: Optional[float] = None,
                 clamp_luts: bool = True,
                 use_cuda: bool = True,
                 regularizers: dict = None):
        """
        Args:
            num_inputs: Number of input bits (n)
            output_dim: Number of output dimensions
            alpha: EFD parameter for input gradients
            beta: EFD parameter for weight gradients
            clamp_luts: Whether to clamp LUT values to [-1, 1] during training
            use_cuda: Whether to use CUDA kernels (if available)
            regularizers: Dict of custom regularization functions
        """
        super().__init__(num_inputs=num_inputs, regularizers=regularizers)
        self.output_dim = output_dim
        self.clamp_luts = clamp_luts
        self.use_cuda = use_cuda and is_cuda_available()
        
        # EFD parameters matching CUDA defaults
        if alpha is None:
            self.alpha = 0.5 * (0.75 ** (num_inputs - 1))
        else:
            self.alpha = float(alpha)
            
        if beta is None:
            self.beta = 0.25 / 0.75
        else:
            self.beta = float(beta)
        
        # Initialize LUT weights: shape (output_dim, 2^num_inputs)
        lut_size = 2 ** num_inputs
        self.luts = nn.Parameter(
            torch.rand(output_dim, lut_size) * 2 - 1  # Uniform [-1, 1]
        )
        
        # Create mapping tensor (each LUT maps to all inputs in order)
        # Shape: (output_dim, num_inputs)
        self.register_buffer('mapping', torch.arange(num_inputs, dtype=torch.int32).unsqueeze(0).expand(output_dim, -1))
         
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
        Forward pass during training with optional CUDA acceleration.
        """
        # Handle dimension
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # Clamp LUTs if needed
        if self.clamp_luts:
            with torch.no_grad():
                self.luts.data.clamp_(-1, 1)
        
        # Use CUDA kernels if available and on GPU
        if self.use_cuda and x.is_cuda:
            # Ensure input is contiguous and float32
            x_cont = x.contiguous().float()
            
            # CUDA kernel expects mapping as int32
            mapping = self.mapping.int()
            
            # Call CUDA kernel
            output = efd_forward(x_cont, mapping, self.luts, self.alpha, self.beta)
            
            return output
        else:
            # Fallback to Python implementation
            return self._forward_python(x)
    
    def _forward_python(self, x: torch.Tensor) -> torch.Tensor:
        """
        Python fallback implementation for forward pass.
        """
        # Binary threshold at 0.5 for [0, 1] inputs
        # This handles both binary {0, 1} and continuous [0, 1] encodings
        x_binary = (x > 0.5).float()
        
        # Get indices 
        indices = self._binary_to_index(x_binary)
        
        # Look up values
        if self.output_dim == 1:
            output = self.luts[0, indices]
        else:
            outputs = []
            for dim in range(self.output_dim):
                outputs.append(self.luts[dim, indices])
            output = torch.stack(outputs, dim=1)
        
        return output
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during evaluation (uses same logic as training).
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # Use CUDA for evaluation too if available
        if self.use_cuda and x.is_cuda:
            x_cont = x.contiguous().float()
            mapping = self.mapping.int()
            output = efd_forward(x_cont, mapping, self.luts, self.alpha, self.beta)
            return output
        else:
            return self._forward_python(x)
    
    # Note: backward() is handled by CUDA kernel via EFDFunction in cuda/__init__.py
    # No need for custom backward method here - the CUDA implementation is much faster

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization to match base CUDA implementation."""
        return torch.tensor(0.0, device=self.luts.device, requires_grad=False)