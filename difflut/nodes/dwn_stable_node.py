from typing import Optional, Callable
import warnings
import torch
import torch.nn as nn

from .base_node import BaseNode
from ..registry import register_node
from .cuda import is_cuda_available

# Try to import the compiled CUDA extension
try:
    import dwn_stable_cuda as _dwn_stable_cuda_module
    _CUDA_EXT_AVAILABLE = True
except ImportError:
    _CUDA_EXT_AVAILABLE = False
    _dwn_stable_cuda_module = None
    warnings.warn(
        "CUDA extension 'dwn_stable_cuda' not available. DWNStableNode will use slower CPU fallback. "
        "For better performance, compile the CUDA extension using: "
        "'cd difflut && python setup.py install'. "
        "To suppress this warning: warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut.nodes.dwn_stable_node')",
        RuntimeWarning,
        stacklevel=2
    )


class GradientStabilizedFunction(torch.autograd.Function):
    """
    PyTorch autograd function wrapper for Gradient Stabilized CUDA kernels.
    Handles 3D tensors with per-layer-node parameters.
    Same as EFD but with gradient scaling in backward pass.
    """
    @staticmethod
    def forward(ctx, input, luts, gradient_scale):
        """
        Forward pass using CUDA kernel.
        
        Args:
            input: (batch_size, layer_size, input_dim) float tensor in [0, 1]
            luts: (layer_size, output_dim, 2^input_dim) float tensor in [0, 1]
            gradient_scale: scalar float for gradient scaling
        
        Returns:
            output: (batch_size, layer_size, output_dim) float tensor
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Please compile dwn_stable_cuda extension.")
        
        # Ensure correct dtypes and contiguity
        input = input.contiguous().float()
        luts = luts.contiguous().float()
        
        # Call CUDA forward kernel
        output = _dwn_stable_cuda_module.forward(input, luts)
        
        # Save for backward
        ctx.save_for_backward(input, luts)
        ctx.gradient_scale = gradient_scale
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using CUDA kernel with gradient scaling.
        
        Args:
            grad_output: (batch_size, layer_size, output_dim) gradient tensor
        
        Returns:
            Gradients for (input, luts, gradient_scale)
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Please compile dwn_stable_cuda extension.")
        
        input, luts = ctx.saved_tensors
        gradient_scale = ctx.gradient_scale
        
        # Ensure contiguity
        grad_output = grad_output.contiguous().float()
        
        # Call CUDA backward kernel with gradient scaling
        grad_input, grad_luts = _dwn_stable_cuda_module.backward(
            input, luts, grad_output, gradient_scale
        )
        
        # Return gradients (None for gradient_scale)
        return grad_input, grad_luts, None


class GradientStabilizedFunctionCPU(torch.autograd.Function):
    """
    CPU fallback for Gradient Stabilized with EFD backward pass and gradient scaling.
    Handles 3D tensors with per-layer-node parameters.
    """
    @staticmethod
    def forward(ctx, input, luts, gradient_scale):
        """Forward pass with binary thresholding."""
        # Binary threshold at 0.5 for [0, 1] inputs
        x_binary = (input >= 0.5).float()
        batch_size, layer_size, input_dim = x_binary.shape
        output_dim = luts.shape[1]
        
        # Compute LUT indices from binary inputs
        powers = 2 ** torch.arange(input_dim, device=input.device, dtype=torch.float32)
        indices = (x_binary * powers).sum(dim=-1).long()  # (batch_size, layer_size)
        
        # Look up LUT values: luts is (layer_size, output_dim, 2^input_dim)
        batch_indices = torch.arange(batch_size, device=input.device).view(-1, 1, 1).expand(-1, layer_size, output_dim)
        layer_indices = torch.arange(layer_size, device=input.device).view(1, -1, 1).expand(batch_size, -1, output_dim)
        output_indices = torch.arange(output_dim, device=input.device).view(1, 1, -1).expand(batch_size, layer_size, -1)
        lut_indices = indices.unsqueeze(-1).expand(-1, -1, output_dim)
        
        output = luts[layer_indices, output_indices, lut_indices]  # (batch_size, layer_size, output_dim)
        
        # Save for backward
        ctx.save_for_backward(input, luts)
        ctx.gradient_scale = gradient_scale
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using Gradient Stabilized EFD with gradient scaling."""
        input, luts = ctx.saved_tensors
        gradient_scale = ctx.gradient_scale
        
        batch_size, layer_size, input_dim = input.shape
        output_dim = luts.shape[1]
        lut_size = 2 ** input_dim
        
        grad_input = torch.zeros_like(input)
        grad_luts = torch.zeros_like(luts)
        
        for batch_idx in range(batch_size):
            for layer_idx in range(layer_size):
                # Compute current address from binary input
                addr = 0
                for i in range(input_dim):
                    if input[batch_idx, layer_idx, i].item() >= 0.5:
                        addr |= (1 << i)
                
                # LUT gradient - direct assignment to accessed entry
                for dim_idx in range(output_dim):
                    grad_luts[layer_idx, dim_idx, addr] += grad_output[batch_idx, layer_idx, dim_idx] * gradient_scale
                
                # Input gradient using Gradient Stabilized EFD
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
                            lut_value = luts[layer_idx, dim_idx, k].item()
                            
                            # Gradient stabilized formula: sign * lut / (hamming_dist + 1)
                            total_gradient += (sign_factor * lut_value / (hamming_dist + 1.0) * 
                                             gradient_scale * 
                                             grad_output[batch_idx, layer_idx, dim_idx].item())
                    
                    grad_input[batch_idx, layer_idx, input_idx] = total_gradient
        
        return grad_input, grad_luts, None


def dwn_stable_forward(input, luts, gradient_scale):
    """
    Gradient Stabilized forward pass with automatic differentiation support.
    
    Args:
        input: (batch_size, layer_size, input_dim) tensor in [0, 1]
        luts: (layer_size, output_dim, 2^input_dim) tensor in [0, 1]
        gradient_scale: scalar float for gradient scaling
    
    Returns:
        output: (batch_size, layer_size, output_dim) tensor
    """
    if _CUDA_EXT_AVAILABLE and input.is_cuda:
        return GradientStabilizedFunction.apply(input, luts, gradient_scale)
    else:
        # CPU fallback with proper gradient stabilized EFD backward
        return GradientStabilizedFunctionCPU.apply(input, luts, gradient_scale)

@register_node("dwn_stable")
class DWNStableNode(BaseNode):
    """
    Gradient Stabilized Node - Same as DWN but with gradient scaling.
    
    Forward: Binary thresholding at 0.5 (>= 0.5) for inputs in [0, 1]
    Backward: Extended Finite Difference (EFD) multiplied by gradient_scale
    
    Weights: Raw weights passed through sigmoid to get LUT values in [0, 1]
    Now supports per-layer-node LUTs for better memory access patterns.
    """
    
    def __init__(self, 
                 input_dim: int | None = None,
                 output_dim: int | None = None,
                 layer_size: int | None = None,
                 use_cuda: bool = True,
                 regularizers: dict | None = None,
                 gradient_scale: float = 1.25,
                 init_fn: Optional[Callable] = None,
                 init_kwargs: dict | None = None):
        """
        Args:
            input_dim: Number of inputs (e.g., 6)
            output_dim: Number of outputs (e.g., 1)
            layer_size: Number of parallel nodes in the layer (e.g., 128)
            use_cuda: Whether to use CUDA kernels (if available)
            regularizers: Dict of custom regularization functions
            gradient_scale: Initial gradient scaling factor (learnable)
            init_fn: Optional initialization function for LUT weights.
                    Signature: init_fn(parameter: torch.Tensor, **init_kwargs) -> None
            init_kwargs: Optional dict of kwargs to pass to the initializer function
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, layer_size=layer_size,
                        regularizers=regularizers, init_fn=init_fn, init_kwargs=init_kwargs)
        self.use_cuda = use_cuda and is_cuda_available()
        
        # Warn if CUDA requested but not available
        if use_cuda and not _CUDA_EXT_AVAILABLE:
            warnings.warn(
                "DWNStableNode: CUDA was requested (use_cuda=True) but CUDA extension is not available. "
                "Using CPU fallback which may be significantly slower. "
                "To enable CUDA: compile the extension with 'cd difflut && python setup.py install'",
                RuntimeWarning,
                stacklevel=2
            )
        
        # Learnable gradient scaling factor
        self.gradient_scale = nn.Parameter(torch.tensor(gradient_scale))
        
        # Initialize raw LUT weights with per-layer-node LUTs
        # Shape: (layer_size, num_outputs, 2^num_inputs)
        lut_size = 2 ** self.num_inputs
        self.raw_luts = nn.Parameter(torch.randn(self.layer_size, self.num_outputs, lut_size) * 0.1)
        
        # Apply initialization to raw_luts if init_fn is provided
        self._apply_init_fn(self.raw_luts, name="raw_luts")
         
    def _get_luts(self) -> torch.Tensor:
        """
        Get actual LUT weights by applying sigmoid to raw weights.
        Maps from (-inf, inf) to [0, 1].
        Returns: (layer_size, num_outputs, 2^num_inputs)
        """
        return torch.sigmoid(self.raw_luts)
    
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training.
        Inputs are in [0, 1], binarized using Heaviside at 0.5.
        Outputs are in [0, 1].
        Uses CUDA kernels when available, otherwise CPU fallback.
        
        Args:
            x: Input tensor (batch_size, layer_size, num_inputs)
        Returns:
            Output tensor (batch_size, layer_size, num_outputs)
        """
        batch_size, layer_size, input_dim = x.shape
        
        # Verify layer_size matches
        if layer_size != self.layer_size:
            raise ValueError(
                f"Input layer_size {layer_size} does not match node's layer_size {self.layer_size}"
            )
        
        # Get actual LUT weights via sigmoid: (layer_size, num_outputs, 2^num_inputs)
        luts = self._get_luts()
        
        # Use CUDA if available and requested
        if self.use_cuda and x.is_cuda and _CUDA_EXT_AVAILABLE:
            output = dwn_stable_forward(x, luts, self.gradient_scale.item())
        else:
            # CPU fallback
            output = dwn_stable_forward(x, luts, self.gradient_scale.item())
        
        return output
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Inputs already binarized in {0, 1}.
        Output binarized to {0, 1} using Heaviside at 0.5.
        
        Args:
            x: Input tensor (batch_size, layer_size, num_inputs)
        Returns:
            Output tensor (batch_size, layer_size, num_outputs)
        """
        batch_size, layer_size, input_dim = x.shape
        
        # Verify layer_size matches
        if layer_size != self.layer_size:
            raise ValueError(
                f"Input layer_size {layer_size} does not match node's layer_size {self.layer_size}"
            )
        
        # Get actual LUT weights via sigmoid: (layer_size, num_outputs, 2^num_inputs)
        luts = self._get_luts()
        
        # Inputs are already binarized in {0, 1}, use directly
        x_binary = x.float()
        
        # Compute LUT indices from binary inputs
        powers = 2 ** torch.arange(self.num_inputs, device=x.device, dtype=torch.float32)
        indices = (x_binary * powers).sum(dim=-1).long()  # (batch_size, layer_size)
        
        # Look up LUT values
        batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1, 1).expand(-1, layer_size, self.num_outputs)
        layer_indices = torch.arange(layer_size, device=x.device).view(1, -1, 1).expand(batch_size, -1, self.num_outputs)
        output_indices = torch.arange(self.num_outputs, device=x.device).view(1, 1, -1).expand(batch_size, layer_size, -1)
        lut_indices = indices.unsqueeze(-1).expand(-1, -1, self.num_outputs)
        
        output = luts[layer_indices, output_indices, lut_indices]  # (batch_size, layer_size, num_outputs)
        
        # Binarize output: [0, 1] -> {0, 1} using Heaviside at 0.5
        output = (output >= 0.5).float()
        
        return output
    
    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization to match base CUDA implementation."""
        return torch.tensor(0.0, device=self.raw_luts.device, requires_grad=False)
