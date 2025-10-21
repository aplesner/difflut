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
    def forward(ctx, input, mapping, luts, alpha, beta):
        """
        Forward pass using CUDA kernel.
        
        Args:
            input: (batch_size, input_length) float tensor in [-1, 1]
            mapping: (num_luts, n) int tensor - mapping of inputs to each LUT
            luts: (num_luts, 2^n) float tensor in [-1, 1] - LUT values
            alpha: scalar tensor for gradient scaling
            beta: scalar tensor for Hamming distance decay
        
        Returns:
            output: (batch_size, num_luts) float tensor in [-1, 1]
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
        ctx.save_for_backward(input, mapping, luts, alpha, beta)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using CUDA kernel with alpha/beta scaling.
        
        Args:
            grad_output: (batch_size, num_luts) gradient tensor
        
        Returns:
            Gradients for (input, mapping, luts, alpha, beta)
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Please compile efd_cuda extension.")
        
        input, mapping, luts, alpha, beta = ctx.saved_tensors
        
        # Ensure contiguity
        grad_output = grad_output.contiguous().float()
        
        # Call CUDA backward kernel with alpha and beta
        grad_input, grad_luts = _efd_cuda_module.backward(
            input, mapping, luts, grad_output, alpha.item(), beta.item()
        )
        
        # Return gradients (None for mapping, alpha, beta)
        return grad_input, None, grad_luts, None, None


class EFDFunctionCPU(torch.autograd.Function):
    """
    CPU fallback for EFD with proper Extended Finite Difference backward pass.
    """
    @staticmethod
    def forward(ctx, input, mapping, luts, alpha, beta):
        """Forward pass with binary thresholding."""
        # Binary threshold at 0.0 for [-1, 1] inputs
        x_binary = (input >= 0.0).float()
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
        ctx.save_for_backward(input, mapping, luts, alpha, beta)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using Extended Finite Difference (EFD) with alpha/beta scaling."""
        input, mapping, luts, alpha, beta = ctx.saved_tensors
        batch_size = input.shape[0]
        input_length = input.shape[1]
        num_luts = luts.shape[0]
        n = mapping.shape[1]
        lut_size = 2 ** n
        
        grad_input = torch.zeros_like(input)
        grad_luts = torch.zeros_like(luts)
        
        alpha_val = alpha.item()
        beta_val = beta.item()
        
        for i in range(batch_size):
            for j in range(num_luts):
                # Compute current address (for LUT gradient)
                addr = 0
                for l in range(n):
                    if input[i, mapping[j, l]] >= 0.0:
                        addr |= (1 << l)
                
                # LUT gradient
                grad_luts[j, addr] += grad_output[i, j]
                
                # Input gradient using Extended Finite Difference with alpha/beta scaling
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
                        
                        # Add weighted contribution: alpha * sign * lut * beta^hamming_dist
                        total_gradient += alpha_val * sign_factor * lut_value * (beta_val ** hamming_dist)
                    
                    grad_input[i, mapping[j, l]] += total_gradient * grad_output[i, j]
        
        return grad_input, None, grad_luts, None, None


def efd_forward(input, mapping, luts, alpha, beta):
    """
    EFD forward pass with automatic differentiation support.
    
    Args:
        input: (batch_size, input_length) tensor in [-1, 1]
        mapping: (num_luts, n) int tensor
        luts: (num_luts, 2^n) tensor in [-1, 1]
        alpha: scalar tensor for gradient scaling
        beta: scalar tensor for Hamming distance decay
    
    Returns:
        output: (batch_size, num_luts) tensor in [-1, 1]
    """
    if _CUDA_EXT_AVAILABLE and input.is_cuda:
        return EFDFunction.apply(input, mapping, luts, alpha, beta)
    else:
        # CPU fallback with proper EFD backward
        return EFDFunctionCPU.apply(input, mapping, luts, alpha, beta)

@register_node("dwn")
class DWNNode(BaseNode):
    """
    Differentiable Weightless Neural Network node with Extended Finite Difference (EFD).
    
    Forward: Binary thresholding at 0.0 for inputs in [-1, 1]
    Backward: Extended Finite Difference (EFD) with alpha/beta scaling:
              alpha * (-1)^(1-k_j) * lut[k] * beta^hamming_dist
    
    Weights: LUT values in [-1, 1], clamped during training
    """
    
    def __init__(self, 
                 input_dim: list = None,
                 output_dim: list = None,
                 use_cuda: bool = True,
                 regularizers: dict = None,
                 alpha: float = None,
                 beta: float = None,
                 clamp_luts: bool = True):
        """
        Args:
            input_dim: Input dimensions as list (e.g., [6])
            output_dim: Output dimensions as list (e.g., [1])
            use_cuda: Whether to use CUDA kernels (if available)
            regularizers: Dict of custom regularization functions
            alpha: Gradient scaling factor (default: 0.5 * 0.75^(n-1))
            beta: Hamming distance decay factor (default: 0.25/0.75)
            clamp_luts: Whether to clamp LUT values to [-1, 1] during training
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, regularizers=regularizers)
        self.use_cuda = use_cuda and is_cuda_available()
        self.clamp_luts = clamp_luts
        
        # Warn if CUDA requested but not available
        if use_cuda and not _CUDA_EXT_AVAILABLE:
            warnings.warn(
                "DWNNode: CUDA was requested (use_cuda=True) but CUDA extension is not available. "
                "Using CPU fallback which may be significantly slower. "
                "To enable CUDA: compile the extension with 'cd difflut && python setup.py install'",
                RuntimeWarning,
                stacklevel=2
            )
        
        # Set alpha and beta based on input dimension
        if alpha is None:
            alpha = 0.5 * (0.75 ** (self.num_inputs - 1))
        if beta is None:
            beta = 0.25 / 0.75
        
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer('beta', torch.tensor(beta, dtype=torch.float32))
        
        # Initialize LUT weights: shape (num_outputs, 2^num_inputs)
        # Uniform initialization in [-1, 1]
        lut_size = 2 ** self.num_inputs
        self.luts = nn.Parameter(
            torch.rand(self.num_outputs, lut_size) * 2 - 1  # Uniform in [-1, 1]
        )
        
        # Create mapping tensor (each LUT maps to all inputs in order)
        # Shape: (num_outputs, num_inputs)
        self.register_buffer('mapping', torch.arange(self.num_inputs, dtype=torch.int32).unsqueeze(0).expand(self.num_outputs, -1))
         
    def _clamp_luts_if_needed(self):
        """Clamp LUT values to [-1, 1] during training if enabled."""
        if self.training and self.clamp_luts:
            with torch.no_grad():
                self.luts.clamp_(-1.0, 1.0)
    
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
        Inputs are in [-1, 1], binarized using Heaviside at 0.0.
        Outputs are in [-1, 1].
        """
        x = self._prepare_input(x)
        
        # Clamp LUT values to [-1, 1] if enabled
        self._clamp_luts_if_needed()
        
        # Use efd_forward which handles CUDA/CPU dispatch automatically
        if self.use_cuda and x.is_cuda:
            x = x.contiguous().float()
            mapping = self.mapping.int()
        else:
            mapping = self.mapping
        
        output = efd_forward(x, mapping, self.luts, self.alpha, self.beta)
        return self._prepare_output(output)
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Inputs already binarized in {-1, 1}.
        Output binarized to {-1, 1} using Heaviside at 0.0.
        """
        x = self._prepare_input(x)
        
        # Inputs are already binarized in {-1, 1}, convert to {0, 1} for indexing
        x_binary = (x >= 0.0).float()
        indices = self._binary_to_index(x_binary)
        
        if self.num_outputs == 1:
            output = self.luts[0, indices]
        else:
            outputs = []
            for dim in range(self.num_outputs):
                outputs.append(self.luts[dim, indices])
            output = torch.stack(outputs, dim=1)
        
        # Binarize output: [-1, 1] -> {-1, 1} using Heaviside at 0.0
        output = torch.where(output > 0.0, torch.ones_like(output), -torch.ones_like(output))
        
        return self._prepare_output(output)
    
    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization to match base CUDA implementation."""
        return torch.tensor(0.0, device=self.luts.device, requires_grad=False)