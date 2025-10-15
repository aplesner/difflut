import torch
import torch.nn as nn
from typing import Optional
from .base_node import BaseNode
from ..registry import register_node
from .cuda import is_cuda_available, hybrid_forward

@register_node("hybrid")
class HybridNode(BaseNode):
    """
    Hybrid LUT node combining DWN and UnboundProbabilistic approaches.
    
    Forward pass: Uses binary thresholding like DWN (discrete, efficient)
    Backward pass: Uses probabilistic gradients like UnboundProbabilistic (smooth, trainable)
    
    This combines the best of both worlds:
    - Fast, discrete inference (DWN)
    - Smooth, effective gradients (UnboundProbabilistic)
    """
    
    def __init__(self, 
                 num_inputs: int, 
                 output_dim: int = 1,
                 use_cuda: bool = True,
                 regularizers: dict = None):
        """
        Args:
            num_inputs: Number of input bits (n)
            output_dim: Number of output dimensions
            use_cuda: Whether to use CUDA kernels (if available)
            regularizers: Dict of custom regularization functions
        """
        super().__init__(num_inputs=num_inputs, regularizers=regularizers)
        self.output_dim = output_dim
        self.use_cuda = use_cuda and is_cuda_available()
        
        # Initialize LUT weights: shape (output_dim, 2^num_inputs)
        lut_size = 2 ** num_inputs
        self.luts = nn.Parameter(
            torch.rand(output_dim, lut_size) * 2 - 1  # Uniform [-1, 1]
        )
        
        # Create mapping tensor (each LUT maps to all inputs in order)
        # Shape: (output_dim, num_inputs)
        self.register_buffer(
            'mapping', 
            torch.arange(num_inputs, dtype=torch.int32).unsqueeze(0).expand(output_dim, -1)
        )
        
        # Precompute all binary combinations for probabilistic backward
        binary_combinations = []
        for i in range(2**num_inputs):
            bits = []
            for j in range(num_inputs):
                bits.append((i >> j) & 1)
            binary_combinations.append(bits)
        
        self.register_buffer(
            'binary_combinations',
            torch.tensor(binary_combinations, dtype=torch.float32)
        )
         
    def _binary_to_index(self, x_binary: torch.Tensor) -> torch.Tensor:
        """
        Convert binary input to LUT index.
        
        Args:
            x_binary: Binary tensor of shape (batch_size, num_inputs)
        Returns:
            Indices of shape (batch_size,)
        """
        batch_size = x_binary.shape[0]
        device = x_binary.device
        indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Build index bit by bit (LSB first, matching DWN)
        for l in range(self.num_inputs):
            indices = indices | (x_binary[:, l].long() << l)
        
        return indices
    
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training with hybrid behavior.
        Uses CUDA kernel if available, otherwise falls back to Python.
        """
        # Handle dimension
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # Use CUDA kernels if available and on GPU
        if self.use_cuda and x.is_cuda:
            # Ensure input is contiguous and float32
            x_cont = x.contiguous().float()
            
            # CUDA kernel expects mapping as int32
            mapping = self.mapping.int()
            
            # Call hybrid CUDA kernel
            output = hybrid_forward(
                x_cont, 
                mapping, 
                self.luts,
                self.binary_combinations
            )
            
            return output
        else:
            # Fallback to Python implementation
            return self._forward_python(x)
    
    def _forward_python(self, x: torch.Tensor) -> torch.Tensor:
        """
        Python fallback implementation with custom backward.
        Forward: Binary thresholding (DWN-style)
        Backward: Probabilistic gradients (UnboundProbabilistic-style)
        """
        # Use custom autograd function
        return HybridFunction.apply(x, self.luts, self.binary_combinations, self.num_inputs, self.output_dim)
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during evaluation (same as training forward).
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # Use CUDA for evaluation too if available
        if self.use_cuda and x.is_cuda:
            x_cont = x.contiguous().float()
            mapping = self.mapping.int()
            output = hybrid_forward(x_cont, mapping, self.luts, self.binary_combinations)
            return output
        else:
            return self._forward_python(x)

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization."""
        return torch.tensor(0.0, device=self.luts.device, requires_grad=False)


class HybridFunction(torch.autograd.Function):
    """
    Custom autograd function for hybrid forward/backward.
    Forward: Binary thresholding (discrete)
    Backward: Probabilistic gradients (continuous)
    """
    
    @staticmethod
    def forward(ctx, x, luts, binary_combinations, num_inputs, output_dim):
        """
        Forward pass: Binary thresholding like DWN.
        
        Args:
            x: Input tensor (batch_size, num_inputs)
            luts: LUT weights (output_dim, 2^num_inputs)
            binary_combinations: Precomputed binary patterns (2^num_inputs, num_inputs)
            num_inputs: Number of inputs
            output_dim: Number of output dimensions
        """
        # Save for backward
        ctx.save_for_backward(x, luts, binary_combinations)
        ctx.num_inputs = num_inputs
        ctx.output_dim = output_dim
        
        # Forward: Binary thresholding at 0.5 for [0, 1] inputs
        x_binary = (x > 0.5).float()
        
        # Convert to indices
        batch_size = x.shape[0]
        device = x.device
        indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for l in range(num_inputs):
            indices = indices | (x_binary[:, l].long() << l)
        
        # Look up values
        if output_dim == 1:
            output = luts[0, indices]
        else:
            outputs = []
            for dim in range(output_dim):
                outputs.append(luts[dim, indices])
            output = torch.stack(outputs, dim=1)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Probabilistic gradients like UnboundProbabilistic.
        
        This provides smooth gradients even though forward was discrete.
        """
        x, luts, binary_combinations = ctx.saved_tensors
        num_inputs = ctx.num_inputs
        output_dim = ctx.output_dim
        
        batch_size = x.shape[0]
        
        # Ensure x is in [0, 1] range for probabilistic computation
        # Apply sigmoid to map any range to [0, 1]
        x_prob = torch.sigmoid(x)
        
        # Compute probabilistic expectation for gradient
        # x_prob: (batch_size, num_inputs)
        # binary_combinations: (2^num_inputs, num_inputs)
        
        # Expand for broadcasting
        x_expanded = x_prob.unsqueeze(1)  # (batch_size, 1, num_inputs)
        a_expanded = binary_combinations.unsqueeze(0)  # (1, 2^num_inputs, num_inputs)
        
        # Compute Pr(a|x) = prod_j [x_j^a_j * (1-x_j)^(1-a_j)]
        prob_terms = x_expanded * a_expanded + (1 - x_expanded) * (1 - a_expanded)
        probs = torch.prod(prob_terms, dim=2)  # (batch_size, 2^num_inputs)
        
        # Gradient w.r.t. LUTs
        # grad_output: (batch_size,) or (batch_size, output_dim)
        if grad_output.dim() == 1:
            grad_output = grad_output.unsqueeze(1)
        
        # grad_luts = probs^T @ grad_output
        # probs: (batch_size, 2^num_inputs)
        # grad_output: (batch_size, output_dim)
        # Result: (2^num_inputs, output_dim)
        grad_luts = torch.matmul(probs.t(), grad_output)
        grad_luts = grad_luts.t()  # (output_dim, 2^num_inputs)
        
        # Gradient w.r.t. inputs
        # For each input j, ∂Pr(a|x)/∂x_j = Pr(a|x) * (a_j - x_j) / (x_j * (1 - x_j))
        grad_input = torch.zeros_like(x)
        
        for j in range(num_inputs):
            # Derivative of probability w.r.t. x_j
            # ∂Pr(a|x)/∂x_j = Pr(a|x) * [(a_j - x_j) / (x_j(1-x_j) + eps)]
            eps = 1e-8
            a_j = binary_combinations[:, j].unsqueeze(0)  # (1, 2^num_inputs)
            x_j = x_prob[:, j].unsqueeze(1)  # (batch_size, 1)
            
            # Compute derivative factor
            deriv_factor = (a_j - x_j) / (x_j * (1 - x_j) + eps)
            
            # Weight by probability
            prob_deriv = probs * deriv_factor  # (batch_size, 2^num_inputs)
            
            # Sum over LUT entries weighted by LUT values and output gradient
            for dim in range(output_dim):
                lut_weights = luts[dim, :].unsqueeze(0)  # (1, 2^num_inputs)
                grad_j = torch.sum(prob_deriv * lut_weights, dim=1)  # (batch_size,)
                
                if output_dim == 1:
                    grad_input[:, j] += grad_j * grad_output[:, dim]
                else:
                    grad_input[:, j] += grad_j * grad_output[:, dim]
            
            # Apply chain rule for sigmoid
            sigmoid_grad = x_prob[:, j] * (1 - x_prob[:, j])
            grad_input[:, j] *= sigmoid_grad
        
        return grad_input, grad_luts, None, None, None


# Export
__all__ = ['HybridNode']
