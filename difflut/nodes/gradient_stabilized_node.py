from typing import Optional
import torch
import torch.nn as nn
from .base_node import BaseNode
from ..registry import register_node
from .cuda import is_cuda_available

# Try to import the Gradient Stabilized CUDA extension
try:
    import gradient_stabilized_cuda as _gradient_stabilized_cuda_module
    _GRADIENT_STABILIZED_CUDA_EXT_AVAILABLE = True
except ImportError:
    _GRADIENT_STABILIZED_CUDA_EXT_AVAILABLE = False
    _gradient_stabilized_cuda_module = None


class GradientStabilizedFunctionCUDA(torch.autograd.Function):
    """
    Gradient-Stabilized function using CUDA kernels.
    Forward: Binary thresholding (DWN-style)
    Backward: Scaled gradients with input-dependent bounds
    """
    
    @staticmethod
    def forward(ctx, input, mapping, luts, gradient_scale):
        """
        Forward pass using CUDA: Binary thresholding at 0.5
        
        Args:
            input: (batch_size, input_length) tensor in [0, 1]
            mapping: (num_luts, n) int tensor
            luts: (num_luts, 2^n) tensor in [0, 1]
            gradient_scale: scalar tensor for gradient scaling
        """
        output = _gradient_stabilized_cuda_module.forward(input, mapping, luts)
        ctx.save_for_backward(input, mapping, luts, gradient_scale)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using CUDA: Simplified gradient with scaling and bounds
        
        Creates smooth gradients near decision boundary (0.5),
        zero gradients far from boundary.
        """
        input, mapping, luts, gradient_scale = ctx.saved_tensors
        grad_scale_value = gradient_scale.item() if gradient_scale.numel() == 1 else gradient_scale
        
        grad_input, grad_luts = _gradient_stabilized_cuda_module.backward(
            input, mapping, luts, grad_output, float(grad_scale_value)
        )
        
        # Return gradients (None for mapping, gradient_scale gets gradient)
        return grad_input, None, grad_luts, None


class GradientStabilizedFunctionCPU(torch.autograd.Function):
    """
    Gradient-Stabilized EFD with simplified gradient scaling (CPU fallback).
    Forward: Binary thresholding (DWN-style)
    Backward: Scaled gradients with input-dependent bounds
    """
    
    @staticmethod
    def forward(ctx, input, mapping, luts, gradient_scale):
        """
        Forward pass: Binary thresholding at 0.5
        
        Args:
            input: (batch_size, input_length) tensor in [0, 1]
            mapping: (num_luts, n) int tensor
            luts: (num_luts, 2^n) tensor in [0, 1]
            gradient_scale: scalar tensor for gradient scaling
        """
        # Binary thresholding at 0.5
        x_binary = (input >= 0.5).float()
        batch_size = x_binary.shape[0]
        num_luts = luts.shape[0]
        n = mapping.shape[1]
        
        output = torch.zeros(batch_size, num_luts, device=input.device, dtype=input.dtype)
        
        # Compute LUT lookups
        for i in range(batch_size):
            for j in range(num_luts):
                addr = 0
                for l in range(n):
                    if x_binary[i, mapping[j, l]] > 0:
                        addr |= (1 << l)
                output[i, j] = luts[j, addr]
        
        ctx.save_for_backward(input, mapping, luts, gradient_scale)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Simplified gradient with scaling and bounds
        
        Creates smooth gradients near decision boundary (0.5),
        zero gradients far from boundary.
        """
        input, mapping, luts, gradient_scale = ctx.saved_tensors
        batch_size = input.shape[0]
        input_length = input.shape[1]
        num_luts = luts.shape[0]
        n = mapping.shape[1]
        
        grad_input = torch.zeros_like(input)
        grad_luts = torch.zeros_like(luts)
        
        # Apply gradient scaling
        scaled_grad_output = grad_output * gradient_scale
        
        for i in range(batch_size):
            for j in range(num_luts):
                # Current address
                addr = 0
                for l in range(n):
                    if input[i, mapping[j, l]] >= 0.5:
                        addr |= (1 << l)
                
                # LUT gradient (standard accumulation)
                grad_luts[j, addr] += scaled_grad_output[i, j]
                
                # INPUT GRADIENT: Linear interpolation based on distance from threshold
                for l in range(n):
                    input_idx = mapping[j, l]
                    x_val = input[i, input_idx]
                    
                    # Compute gradient based on input distance from threshold
                    # Distance from 0.5: ranges from 0 (at extremes) to 1 (at 0.5)
                    distance_from_threshold = 1.0 - 2.0 * torch.abs(x_val - 0.5)
                    distance_from_threshold = torch.clamp(distance_from_threshold, 0.0, 1.0)
                    
                    # Gradient magnitude based on LUT variation
                    lut_variation = luts[j].max() - luts[j].min()
                    gradient_magnitude = distance_from_threshold * lut_variation
                    
                    # Apply gradient with proper sign based on which side of threshold
                    if x_val >= 0.5:
                        grad_input[i, input_idx] += gradient_magnitude * scaled_grad_output[i, j]
                    else:
                        grad_input[i, input_idx] -= gradient_magnitude * scaled_grad_output[i, j]
        
        # Return gradients (None for mapping, gradient_scale gets gradient)
        return grad_input, None, grad_luts, None


def gradient_stabilized_forward(input, mapping, luts, gradient_scale, use_cuda=True):
    """
    Gradient-stabilized forward pass with automatic differentiation support.
    
    Args:
        input: (batch_size, input_length) tensor in [0, 1]
        mapping: (num_luts, n) int tensor
        luts: (num_luts, 2^n) tensor in [0, 1]
        gradient_scale: scalar tensor for gradient scaling
        use_cuda: whether to use CUDA implementation
    
    Returns:
        output: (batch_size, num_luts) tensor
    """
    # Use CUDA if available and requested
    if use_cuda and input.is_cuda and _GRADIENT_STABILIZED_CUDA_EXT_AVAILABLE:
        return GradientStabilizedFunctionCUDA.apply(input, mapping, luts, gradient_scale)
    else:
        return GradientStabilizedFunctionCPU.apply(input, mapping, luts, gradient_scale)


@register_node("gradient_stabilized")
class GradientStabilizedNode(BaseNode):
    """
    Gradient-Stabilized LUT node with simplified gradient scaling.
    
    Forward pass: Binary thresholding (DWN-style)
    Backward pass: Scaled gradients with input-dependent bounds
    
    Key features:
    - Smooth gradients near decision boundary (0.5)
    - Zero gradients far from boundary
    - Learnable gradient scaling factor
    - LUT variation-based gradient magnitude
    """
    
    def __init__(self, 
                 input_dim: list = None,
                 output_dim: list = None,
                 use_cuda: bool = True,
                 regularizers: dict = None,
                 gradient_scale: float = 1.0):
        """
        Args:
            input_dim: Input dimensions as list (e.g., [6])
            output_dim: Output dimensions as list (e.g., [1])
            use_cuda: Whether to use CUDA kernels (if available)
            regularizers: Dict of custom regularization functions
            gradient_scale: Initial gradient scaling factor (learnable)
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, regularizers=regularizers)
        self.use_cuda = use_cuda and is_cuda_available()
        
        # Learnable gradient scaling factor
        self.gradient_scale = nn.Parameter(torch.tensor(gradient_scale))
        
        # Initialize raw LUT weights: shape (num_outputs, 2^num_inputs)
        # Initialize near decision boundary with small noise
        lut_size = 2 ** self.num_inputs
        self.raw_luts = nn.Parameter(
            torch.randn(self.num_outputs, lut_size) * 0.05
        )
        
        # Create mapping tensor (each LUT maps to all inputs in order)
        # Shape: (num_outputs, num_inputs)
        self.register_buffer(
            'mapping', 
            torch.arange(self.num_inputs, dtype=torch.int32).unsqueeze(0).expand(self.num_outputs, -1)
        )
    
    def _get_luts(self) -> torch.Tensor:
        """
        Get actual LUT weights by applying sigmoid to raw weights.
        Maps from (-inf, inf) to [0, 1].
        """
        return torch.sigmoid(self.raw_luts)
    
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training with gradient-stabilized backward.
        Inputs are in [0, 1], binarized using Heaviside at 0.5.
        """
        x = self._prepare_input(x)
        
        # Get actual LUT weights via sigmoid
        luts = self._get_luts()
        
        # Use gradient-stabilized forward with CUDA support
        output = gradient_stabilized_forward(x, self.mapping, luts, self.gradient_scale, self.use_cuda)
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
        x_binary = x.long()
        batch_size = x_binary.shape[0]
        device = x_binary.device
        
        indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        for l in range(self.num_inputs):
            indices = indices | (x_binary[:, l].long() << l)
        
        if self.num_outputs == 1:
            output = luts[0, indices]
        else:
            outputs = []
            for dim in range(self.num_outputs):
                outputs.append(luts[dim, indices])
            output = torch.stack(outputs, dim=1)
        
        # Binarize output: [0, 1] -> {0, 1} using Heaviside at 0.5
        output = (output >= 0.5).float()
        
        return self._prepare_output(output)

    def _builtin_regularization(self) -> torch.Tensor:
        """
        Built-in regularization for gradient-stabilized node.
        Encourages gradient_scale to stay reasonable.
        """
        # Penalize extreme gradient scales
        scale_penalty = torch.abs(self.gradient_scale - 1.0)
        return 0.01 * scale_penalty


# Export
__all__ = ['GradientStabilizedNode']
