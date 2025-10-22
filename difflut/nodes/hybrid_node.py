import torch
import torch.nn as nn
from typing import Optional, Callable
import warnings
from .base_node import BaseNode
from ..registry import register_node
from .cuda import is_cuda_available

# Try to import the hybrid CUDA extension
try:
    import hybrid_cuda as _hybrid_cuda_module
    _HYBRID_CUDA_EXT_AVAILABLE = True
except ImportError:
    _HYBRID_CUDA_EXT_AVAILABLE = False
    _hybrid_cuda_module = None
    warnings.warn(
        "CUDA extension 'hybrid_cuda' not available. HybridNode will use slower CPU fallback. "
        "For better performance, compile the CUDA extension using: "
        "'cd difflut && python setup.py install'. "
        "To suppress this warning: warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut.nodes.hybrid_node')",
        RuntimeWarning,
        stacklevel=2
    )


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


class HybridFunctionCPU(torch.autograd.Function):
    """
    CPU fallback for Hybrid with custom backward.
    Forward: Binary thresholding at 0.5 (DWN-style)
    Backward: Probabilistic gradients (UnboundProbabilistic-style)
    """
    
    @staticmethod
    def forward(ctx, x, luts, binary_combinations, num_inputs, output_dim):
        """
        Forward pass: Binary thresholding like DWN.
        
        Args:
            x: Input tensor (batch_size, num_inputs) in [0, 1]
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
        x_binary = (x >= 0.5).float()
        
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
        
        # Input x is already in [0, 1] range, use directly for probabilistic computation
        x_prob = x
        
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
        if grad_output.dim() == 1:
            grad_output = grad_output.unsqueeze(1)
        
        # grad_luts = probs^T @ grad_output
        grad_luts = torch.matmul(probs.t(), grad_output)
        grad_luts = grad_luts.t()  # (output_dim, 2^num_inputs)
        
        # Gradient w.r.t. inputs
        grad_input = torch.zeros_like(x)
        
        for j in range(num_inputs):
            # Derivative of probability w.r.t. x_j
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
                grad_input[:, j] += grad_j * grad_output[:, dim]
        
        return grad_input, grad_luts, None, None, None


def hybrid_forward(input, mapping, luts, binary_combinations):
    """
    Hybrid forward pass with automatic differentiation support.
    Forward: Binary thresholding at 0.5 (efficient, discrete)
    Backward: Probabilistic gradients (smooth, trainable)
    
    Args:
        input: (batch_size, input_length) tensor in [0, 1]
        mapping: (num_luts, n) int tensor
        luts: (num_luts, 2^n) tensor in [0, 1]
        binary_combinations: (2^n, n) tensor - precomputed binary patterns
    
    Returns:
        output: (batch_size, num_luts) tensor
    """
    if _HYBRID_CUDA_EXT_AVAILABLE and input.is_cuda:
        return HybridFunction.apply(input, mapping, luts, binary_combinations)
    else:
        # CPU fallback - extract num_inputs and output_dim from shapes
        num_inputs = mapping.shape[1]
        output_dim = luts.shape[0]
        return HybridFunctionCPU.apply(input, luts, binary_combinations, num_inputs, output_dim)

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
                 input_dim: list = None,
                 output_dim: list = None,
                 use_cuda: bool = True,
                 regularizers: dict = None,
                 init_fn: Optional[Callable] = None):
        """
        Args:
            input_dim: Input dimensions as list (e.g., [6])
            output_dim: Output dimensions as list (e.g., [1])
            use_cuda: Whether to use CUDA kernels (if available)
            regularizers: Dict of custom regularization functions
            init_fn: Optional initialization function for LUT weights
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, regularizers=regularizers, init_fn=init_fn)
        self.use_cuda = use_cuda and is_cuda_available()
        
        # Initialize raw LUT weights: shape (num_outputs, 2^num_inputs)
        lut_size = 2 ** self.num_inputs
        if self.init_fn:
            self.raw_luts = nn.Parameter(self.init_fn((self.num_outputs, lut_size)))
        else:
            # Default: Gaussian initialization around 0
            self.raw_luts = nn.Parameter(torch.randn(self.num_outputs, lut_size) * 0.1)
        
        # Create mapping tensor (each LUT maps to all inputs in order)
        # Shape: (num_outputs, num_inputs)
        self.register_buffer(
            'mapping', 
            torch.arange(self.num_inputs, dtype=torch.int32).unsqueeze(0).expand(self.num_outputs, -1)
        )
        
        # Precompute all binary combinations for probabilistic backward
        binary_combinations = []
        for i in range(2**self.num_inputs):
            bits = []
            for j in range(self.num_inputs):
                bits.append((i >> j) & 1)
            binary_combinations.append(bits)
        
        self.register_buffer(
            'binary_combinations',
            torch.tensor(binary_combinations, dtype=torch.float32)
        )
    
    def _get_luts(self) -> torch.Tensor:
        """
        Get actual LUT weights by applying sigmoid to raw weights.
        Maps from (-inf, inf) to [0, 1].
        """
        return torch.sigmoid(self.raw_luts)
         
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
        Forward pass during training with automatic CUDA/CPU dispatch.
        Inputs are in [0, 1], binarized using Heaviside at 0.5.
        """
        x = self._prepare_input(x)
        
        # Get actual LUT weights via sigmoid
        luts = self._get_luts()
        
        # Use hybrid_forward which handles CUDA/CPU dispatch automatically
        if self.use_cuda and x.is_cuda:
            x = x.contiguous().float()
            mapping = self.mapping.int()
        else:
            mapping = self.mapping
        
        output = hybrid_forward(x, mapping, luts, self.binary_combinations)
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
            indices = indices | (x_binary[:, l] << l)
        
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
        """No built-in regularization."""
        return torch.tensor(0.0, device=self.raw_luts.device, requires_grad=False)


# Export
__all__ = ['HybridNode']
