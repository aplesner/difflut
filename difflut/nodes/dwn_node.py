import torch
import torch.nn as nn
from typing import Optional, Callable, Tuple
import warnings
from .base_node import BaseNode
from ..registry import register_node
from .cuda import is_cuda_available
from ..utils.warnings import warn_default_value

# Base gradient scaling factor for DWN nodes
# Default alpha = DWN_ALPHA_BASE * (DWN_ALPHA_DECAY ** (n-1))
DWN_ALPHA_BASE: float = 0.5
# Decay factor for alpha based on number of inputs
# Applied as: alpha = 0.5 * (0.75 ** (n-1))
DWN_ALPHA_DECAY: float = 0.75
# Hamming distance decay factor for DWN backward pass
# Default beta = DWN_BETA_NUMERATOR / DWN_BETA_DENOMINATOR
DWN_BETA_NUMERATOR: float = 0.25
DWN_BETA_DENOMINATOR: float = 0.75
# Binary threshold for DWN forward pass
# Inputs >= DWN_BINARY_THRESHOLD are treated as 1, otherwise 0
DWN_BINARY_THRESHOLD: float = 0.5
# Default flag for using CUDA kernels in DWN nodes
DEFAULT_DWN_USE_CUDA: bool = True
# Default flag for clamping LUT values to [0,1] in DWN nodes
DEFAULT_DWN_CLAMP_LUTS: bool = True

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
    Processes 2D tensors.
    """
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, input: torch.Tensor, luts: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
        """
        Forward pass using CUDA kernel.
        
        Args:
            input: (batch_size, input_dim) float tensor in [0, 1]
            luts: (output_dim, 2^input_dim) float tensor in [0, 1]
            alpha: scalar float for gradient scaling
            beta: scalar float for Hamming distance decay
        
        Returns:
            output: (batch_size, output_dim) float tensor in [0, 1]
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Please compile efd_cuda extension.")
        
        # Ensure correct dtypes and contiguity
        input = input.contiguous().float()
        luts = luts.contiguous().float()
        
        # Call CUDA forward kernel
        output = _efd_cuda_module.forward(input, luts)
        
        # Save for backward
        ctx.save_for_backward(input, luts)
        ctx.alpha = alpha
        ctx.beta = beta
        
        return output
    
    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        Backward pass using CUDA kernel with alpha/beta scaling.
        
        Args:
            grad_output: (batch_size, output_dim) gradient tensor
        
        Returns:
            Gradients for (input, luts, alpha, beta)
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Please compile efd_cuda extension.")
        
        input, luts = ctx.saved_tensors
        alpha = ctx.alpha
        beta = ctx.beta
        
        # Ensure contiguity
        grad_output = grad_output.contiguous().float()
        
        # Call CUDA backward kernel with alpha and beta
        grad_input, grad_luts = _efd_cuda_module.backward(
            input, luts, grad_output, alpha, beta
        )
        
        # Return gradients (None for alpha, beta)
        return grad_input, grad_luts, None, None


class EFDFunctionCPU(torch.autograd.Function):
    """
    CPU fallback for EFD with proper Extended Finite Difference backward pass.
    Processes 2D tensors.
    """
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, input: torch.Tensor, luts: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
        """Forward pass with binary thresholding."""
        # Binary threshold at 0.5 for [0, 1] inputs
        x_binary = (input >= 0.5).float()
        batch_size, input_dim = x_binary.shape
        output_dim = luts.shape[0]
        
        # Compute LUT indices from binary inputs
        powers = 2 ** torch.arange(input_dim, device=input.device, dtype=torch.float32)
        indices = (x_binary * powers).sum(dim=-1).long()  # (batch_size,)
        
        # Look up LUT values: luts is (output_dim, 2^input_dim)
        # indices is (batch_size,)
        # We want output[b, o] = luts[o, indices[b]]
        output = luts[:, indices].T  # (batch_size, output_dim)
        
        # Save for backward
        ctx.save_for_backward(input, luts)
        ctx.alpha = alpha
        ctx.beta = beta
        
        return output
    
    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        """Backward pass using Extended Finite Difference (EFD) with alpha/beta scaling."""
        input, luts = ctx.saved_tensors
        alpha = ctx.alpha
        beta = ctx.beta
        
        batch_size, input_dim = input.shape
        output_dim = luts.shape[0]
        lut_size = 2 ** input_dim
        
        grad_input = torch.zeros_like(input)
        grad_luts = torch.zeros_like(luts)
        
        for batch_idx in range(batch_size):
            # Compute current address from binary input
            addr = 0
            for i in range(input_dim):
                if input[batch_idx, i].item() >= DWN_BINARY_THRESHOLD:
                    addr |= (1 << i)
            
            # LUT gradient - direct assignment to accessed entry
            for dim_idx in range(output_dim):
                grad_luts[dim_idx, addr] += grad_output[batch_idx, dim_idx]
            
            # Input gradient using Extended Finite Difference (EFD)
            for input_idx in range(input_dim):
                # Create mask to exclude input_idx-th bit
                mask = ((1 << input_dim) - 1) & ~(1 << input_idx)
                addr_masked = addr & mask
                
                total_gradient = 0.0
                
                # Iterate over all possible addresses k
                for k in range(lut_size):
                    # Calculate Hamming distance between addr and k, excluding input_idx-th bit
                    k_masked = k & mask
                    hamming_dist = bin(addr_masked ^ k_masked).count('1')
                    
                    # Get k_l (input_idx-th bit of k)
                    k_l = (k >> input_idx) & 1
                    
                    # Calculate sign factor: (-1)^(1-k_l)
                    sign_factor = -1.0 if k_l == 0 else 1.0
                    
                    # Get LUT values at position k for all output dimensions
                    for dim_idx in range(output_dim):
                        lut_value = luts[dim_idx, k].item()
                        
                        # Add weighted contribution: alpha * sign * lut * beta^hamming_dist
                        total_gradient += (alpha * sign_factor * lut_value * 
                                         (beta ** hamming_dist) * 
                                         grad_output[batch_idx, dim_idx].item())
                
                grad_input[batch_idx, input_idx] = total_gradient
        
        return grad_input, grad_luts, None, None


def efd_forward(input: torch.Tensor, luts: torch.Tensor, alpha: float, beta: float) -> Optional[torch.Tensor]:
    """
    EFD forward pass with automatic differentiation support.
    
    Args:
        input: (batch_size, input_dim) tensor in [0, 1]
        luts: (output_dim, 2^input_dim) tensor in [0, 1]
        alpha: scalar float for gradient scaling
        beta: scalar float for Hamming distance decay
    
    Returns:
        output: (batch_size, output_dim) tensor in [0, 1]
    """
    if _CUDA_EXT_AVAILABLE and input.is_cuda:
        return EFDFunction.apply(input, luts, alpha, beta)
    else:
        # CPU fallback with proper EFD backward
        return EFDFunctionCPU.apply(input, luts, alpha, beta)

@register_node("dwn")
class DWNNode(BaseNode):
    """
    Differentiable Weightless Neural Network node with Extended Finite Difference (EFD).
    
    Forward: Binary thresholding at 0.5 (> 0.5) for inputs in [0, 1]
    Backward: Extended Finite Difference (EFD) with alpha/beta scaling:
              alpha * (-1)^(1-k_j) * lut[k] * beta^hamming_dist
    
    Weights: LUT values in [0, 1], clamped during training
    Processes 2D tensors: (batch_size, input_dim) -> (batch_size, output_dim)
    """
    
    def __init__(self, 
                 input_dim: Optional[int] = None,
                 output_dim: Optional[int] = None,
                 use_cuda: bool = True,
                 regularizers: Optional[dict] = None,
                 alpha: Optional[float] = None,
                 beta: Optional[float] = None,
                 clamp_luts: bool = True,
                 init_fn: Optional[Callable] = None,
                 init_kwargs: Optional[dict] = None):
        """
        Args:
            input_dim: Number of inputs (e.g., 6)
            output_dim: Number of outputs (e.g., 1)
            use_cuda: Whether to use CUDA kernels (if available)
            regularizers: Dict of custom regularization functions
            alpha: Gradient scaling factor (default: 0.5 * 0.75^(n-1))
            beta: Hamming distance decay factor (default: 0.25/0.75)
            clamp_luts: Whether to clamp LUT values to [0, 1] during training
            init_fn: Optional initialization function for LUT weights. Should take (param: torch.Tensor, **kwargs)
            init_kwargs: Keyword arguments for init_fn
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim,
                         regularizers=regularizers, init_fn=init_fn, init_kwargs=init_kwargs)
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
            alpha = DWN_ALPHA_BASE * (DWN_ALPHA_DECAY ** (self.num_inputs - 1))
            warn_default_value("alpha", alpha, stacklevel=2)
        if beta is None:
            beta = DWN_BETA_NUMERATOR / DWN_BETA_DENOMINATOR
            warn_default_value("beta", beta, stacklevel=2)
        
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer('beta', torch.tensor(beta, dtype=torch.float32))
        
        # Initialize LUT weights
        # Shape: (output_dim, 2^input_dim)
        lut_size = 2 ** self.num_inputs
        self.luts = nn.Parameter(torch.rand(self.num_outputs, lut_size))
        self._apply_init_fn(self.luts, name="luts")
         
    def _clamp_luts_if_needed(self) -> None:
        """Clamp LUT values to [0, 1] during training if enabled."""
        # Do nothing - clamping should be done outside the forward pass
        # to avoid inplace operations that break autograd
        pass
    
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training.
        Inputs are in [0, 1], binarized using Heaviside at 0.5.
        Outputs are in [0, 1].
        Uses CUDA kernels when available, otherwise CPU fallback.
        
        Args:
            x: Tensor of shape (batch_size, input_dim)
        
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        # Clamp LUT values to [0, 1] if enabled
        self._clamp_luts_if_needed()
        
        # Use CUDA if available and requested
        if self.use_cuda and x.is_cuda and _CUDA_EXT_AVAILABLE:
            output = efd_forward(x, self.luts, self.alpha.item(), self.beta.item())
        else:
            # CPU fallback
            output = efd_forward(x, self.luts, self.alpha.item(), self.beta.item())
        
        return output
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Inputs already binarized in {0, 1}.
        Output binarized to {0, 1} using threshold at 0.5.
        
        Args:
            x: Tensor of shape (batch_size, input_dim)
        
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        # Inputs are already binarized in {0, 1}, use directly
        x_binary = x.float()
        
        # Compute LUT indices from binary inputs
        powers = 2 ** torch.arange(self.num_inputs, device=x.device, dtype=torch.float32)
        indices = (x_binary * powers).sum(dim=-1).long()  # (batch_size,)
        
        # Look up LUT values: luts is (output_dim, lut_size)
        # indices is (batch_size,)
        # We want output[b, o] = luts[o, indices[b]]
        output = self.luts[:, indices].T  # (batch_size, output_dim)
        
        # Binarize output: [0, 1] -> {0, 1} using threshold at 0.5
        output = (output >= 0.5).float()
        
        return output
    
    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization to match base CUDA implementation."""
        return torch.tensor(0.0, device=self.luts.device, requires_grad=False)