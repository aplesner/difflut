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
    Same as EFD but with gradient scaling in backward pass.
    """
    @staticmethod
    def forward(ctx, input, mapping, luts, gradient_scale):
        """
        Forward pass using CUDA kernel.
        
        Args:
            input: (batch_size, input_length) float tensor in [0, 1]
            mapping: (num_luts, n) int tensor - mapping of inputs to each LUT
            luts: (num_luts, 2^n) float tensor in [0, 1] - LUT values
            gradient_scale: scalar tensor for gradient scaling
        
        Returns:
            output: (batch_size, num_luts) float tensor
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Please compile dwn_stable_cuda extension.")
        
        # Ensure correct dtypes and contiguity
        input = input.contiguous().float()
        mapping = mapping.contiguous().int()
        luts = luts.contiguous().float()
        
        # Call CUDA forward kernel (same as EFD)
        output = _dwn_stable_cuda_module.forward(input, mapping, luts)
        
        # Save for backward
        ctx.save_for_backward(input, mapping, luts, gradient_scale)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using CUDA kernel with gradient scaling.
        
        Args:
            grad_output: (batch_size, num_luts) gradient tensor
        
        Returns:
            Gradients for (input, mapping, luts, gradient_scale)
        """
        if not _CUDA_EXT_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Please compile dwn_stable_cuda extension.")
        
        input, mapping, luts, gradient_scale = ctx.saved_tensors
        
        # Ensure contiguity
        grad_output = grad_output.contiguous().float()
        
        # Call CUDA backward kernel with gradient scaling
        grad_input, grad_luts = _dwn_stable_cuda_module.backward(
            input, mapping, luts, grad_output, gradient_scale.item()
        )
        
        # Return gradients (None for mapping and gradient_scale)
        return grad_input, None, grad_luts, None


class GradientStabilizedFunctionCPU(torch.autograd.Function):
    """
    CPU fallback for Gradient Stabilized with EFD backward pass and gradient scaling.
    """
    @staticmethod
    def forward(ctx, input, mapping, luts, gradient_scale):
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
                    # Extract the input index from mapping tensor
                    input_idx = mapping[j, l].item()
                    # Use .item() to convert tensor to scalar for comparison
                    if x_binary[i, input_idx].item() >= 0.5:
                        addr |= (1 << l)
                output[i, j] = luts[j, addr]
        
        # Save for backward
        ctx.save_for_backward(input, mapping, luts, gradient_scale)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using Extended Finite Difference (EFD) with gradient scaling."""
        input, mapping, luts, gradient_scale = ctx.saved_tensors
        batch_size = input.shape[0]
        input_length = input.shape[1]
        num_luts = luts.shape[0]
        n = mapping.shape[1]
        lut_size = 2 ** n
        
        grad_input = torch.zeros_like(input)
        grad_luts = torch.zeros_like(luts)
        
        # Apply gradient scaling
        scale = gradient_scale.item()
        
        for i in range(batch_size):
            for j in range(num_luts):
                # Compute current address (for LUT gradient)
                addr = 0
                for l in range(n):
                    # Extract the input index from mapping tensor and use it as scalar index
                    input_idx = int(mapping[j, l].item())
                    if input[i, input_idx].item() >= 0.5:
                        addr |= (1 << l)
                
                # LUT gradient (with scaling)
                grad_luts[j, addr] += grad_output[i, j] * scale
                
                # Input gradient using Extended Finite Difference (with scaling)
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
                    
                    grad_input[i, mapping[j, l]] += total_gradient * grad_output[i, j] * scale
        
        return grad_input, None, grad_luts, None


def dwn_stable_forward(input, mapping, luts, gradient_scale):
    """
    Gradient Stabilized forward pass with automatic differentiation support.
    
    Args:
        input: (batch_size, input_length) tensor in [0, 1]
        mapping: (num_luts, n) int tensor
        luts: (num_luts, 2^n) tensor in [0, 1]
        gradient_scale: scalar tensor for gradient scaling
    
    Returns:
        output: (batch_size, num_luts) tensor
    """
    if _CUDA_EXT_AVAILABLE and input.is_cuda:
        return GradientStabilizedFunction.apply(input, mapping, luts, gradient_scale)
    else:
        # CPU fallback with proper EFD backward
        return GradientStabilizedFunctionCPU.apply(input, mapping, luts, gradient_scale)

@register_node("dwn_stable")
class DWNStableNode(BaseNode):
    """
    Gradient Stabilized Node - Same as DWN but with gradient scaling.
    
    Forward: Binary thresholding at 0.5 (>= 0.5) for inputs in [0, 1]
    Backward: Extended Finite Difference (EFD) multiplied by gradient_scale
    
    Weights: Raw weights passed through sigmoid to get LUT values in [0, 1]
    """
    
    def __init__(self, 
                 input_dim: int = None,
                 output_dim: int = None,
                 use_cuda: bool = True,
                 regularizers: dict = None,
                 gradient_scale: float = 1.25,
                 init_fn: Optional[Callable] = None,
                 init_kwargs: dict = None):
        """
        Args:
            input_dim: Number of inputs (e.g., 6)
            output_dim: Number of outputs (e.g., 1)
            use_cuda: Whether to use CUDA kernels (if available)
            regularizers: Dict of custom regularization functions
            gradient_scale: Initial gradient scaling factor (learnable)
            init_fn: Optional initialization function for LUT weights.
                    Signature: init_fn(parameter: torch.Tensor, **init_kwargs) -> None
            init_kwargs: Optional dict of kwargs to pass to the initializer function
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, regularizers=regularizers, 
                        init_fn=init_fn, init_kwargs=init_kwargs)
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
        
        # Initialize raw LUT weights: shape (num_outputs, 2^num_inputs)
        lut_size = 2 ** self.num_inputs
        self.raw_luts = nn.Parameter(torch.randn(self.num_outputs, lut_size) * 0.1)
        
        # Apply initialization to raw_luts if init_fn is provided
        self._apply_init_fn(self.raw_luts, name="raw_luts")
        
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
        Outputs are in [0, 1].
        
        Args:
            x: Input tensor (batch_size, layer_size, num_inputs)
        Returns:
            Output tensor (batch_size, layer_size, num_outputs)
        """
        batch_size, layer_size, input_dim = x.shape
        # Reshape to (batch_size * layer_size, num_inputs)
        x_flat = x.view(batch_size * layer_size, input_dim)
        
        # Get actual LUT weights via sigmoid
        luts = self._get_luts()
        
        # Use dwn_stable_forward which handles CUDA/CPU dispatch automatically
        if self.use_cuda and x_flat.is_cuda:
            x_flat = x_flat.contiguous().float()
            mapping = self.mapping.int()
        else:
            mapping = self.mapping
        
        output_flat = dwn_stable_forward(x_flat, mapping, luts, self.gradient_scale)
        
        # Reshape back to (batch_size, layer_size, num_outputs)
        output = output_flat.view(batch_size, layer_size, self.num_outputs)
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
        # Reshape to (batch_size * layer_size, num_inputs)
        x_flat = x.view(batch_size * layer_size, input_dim)
        
        # Get actual LUT weights via sigmoid
        luts = self._get_luts()
        
        # Inputs are already binarized in {0, 1}, use directly
        x_binary = x_flat.float()
        indices = self._binary_to_index(x_binary)
        
        if self.num_outputs == 1:
            output_flat = luts[0, indices]  # (batch_size * layer_size,)
            output_flat = output_flat.unsqueeze(1)  # (batch_size * layer_size, 1)
        else:
            outputs = []
            for dim in range(self.num_outputs):
                outputs.append(luts[dim, indices])
            output_flat = torch.stack(outputs, dim=1)  # (batch_size * layer_size, num_outputs)
        
        # Binarize output: [0, 1] -> {0, 1} using Heaviside at 0.5
        # output >= 0.5 -> 1, output < 0.5 -> 0
        output_flat = (output_flat >= 0.5).float()
        
        # Reshape back to (batch_size, layer_size, num_outputs)
        output = output_flat.view(batch_size, layer_size, self.num_outputs)
        return output
    
    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization to match base CUDA implementation."""
        return torch.tensor(0.0, device=self.raw_luts.device, requires_grad=False)
