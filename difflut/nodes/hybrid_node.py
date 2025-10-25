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
    Updated for 3D tensors with per-layer-node parameters (no mapping - dense connectivity)
    """
    @staticmethod
    def forward(ctx, input, luts, binary_combinations):
        """
        Forward pass using CUDA kernel.
        
        Args:
            input: (batch_size, layer_size, input_dim) float tensor
            luts: (layer_size, output_dim, 2^input_dim) float tensor - per-layer-node LUTs
            binary_combinations: (2^input_dim, input_dim) float tensor - precomputed binary patterns
        
        Returns:
            output: (batch_size, layer_size, output_dim) float tensor
        """
        if not _HYBRID_CUDA_EXT_AVAILABLE:
            raise RuntimeError("Hybrid CUDA extension not available. Please compile hybrid_cuda extension.")
        
        # Ensure correct dtypes and contiguity
        input = input.contiguous().float()
        luts = luts.contiguous().float()
        binary_combinations = binary_combinations.contiguous().float()
        
        # Call CUDA forward kernel (no mapping parameter)
        output = _hybrid_cuda_module.forward(input, luts)
        
        # Save for backward
        ctx.save_for_backward(input, luts, binary_combinations)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using CUDA kernel with probabilistic gradients.
        
        Args:
            grad_output: (batch_size, layer_size, output_dim) gradient tensor
        
        Returns:
            Gradients for (input, luts, binary_combinations)
        """
        if not _HYBRID_CUDA_EXT_AVAILABLE:
            raise RuntimeError("Hybrid CUDA extension not available. Please compile hybrid_cuda extension.")
        
        input, luts, binary_combinations = ctx.saved_tensors
        
        # Ensure contiguity
        grad_output = grad_output.contiguous().float()
        
        # Call CUDA backward kernel (no mapping parameter)
        grad_input, grad_luts = _hybrid_cuda_module.backward(
            input, luts, binary_combinations, grad_output
        )
        
        # Return gradients (None for binary_combinations as it doesn't need gradients)
        return grad_input, grad_luts, None


class HybridFunctionCPU(torch.autograd.Function):
    """
    CPU fallback for Hybrid with custom backward.
    Forward: Binary thresholding at 0.5 (DWN-style)
    Backward: Probabilistic gradients (UnboundProbabilistic-style)
    Updated for 3D tensors with per-layer-node parameters (no mapping - dense connectivity)
    """
    
    @staticmethod
    def forward(ctx, x, luts, binary_combinations):
        """
        Forward pass: Binary thresholding like DWN.
        
        Args:
            x: Input tensor (batch_size, layer_size, input_dim) in [0, 1]
            luts: LUT weights (layer_size, output_dim, 2^input_dim) - per-layer-node LUTs
            binary_combinations: Precomputed binary patterns (2^input_dim, input_dim)
        
        Returns:
            output: (batch_size, layer_size, output_dim)
        """
        batch_size, layer_size, input_dim = x.shape
        layer_size_lut, output_dim, lut_size = luts.shape
        
        # Verify layer_size matches
        if layer_size != layer_size_lut:
            raise ValueError(
                f"Input layer_size {layer_size} does not match LUT layer_size {layer_size_lut}"
            )
        
        # Save for backward
        ctx.save_for_backward(x, luts, binary_combinations)
        
        # Forward: Binary thresholding at 0.5 for [0, 1] inputs
        x_binary = (x >= 0.5).float()  # (batch_size, layer_size, input_dim)
        
        # Convert to indices for each (batch, layer) position
        # Compute powers of 2
        powers = 2 ** torch.arange(input_dim, device=x.device, dtype=torch.float32)
        indices = (x_binary * powers).sum(dim=-1).long()  # (batch_size, layer_size)
        
        # Look up values from per-layer-node LUTs
        # We need to select luts[l, o, indices[b, l]] for each b, l, o
        batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1, 1).expand(-1, layer_size, output_dim)
        layer_indices = torch.arange(layer_size, device=x.device).view(1, -1, 1).expand(batch_size, -1, output_dim)
        output_indices = torch.arange(output_dim, device=x.device).view(1, 1, -1).expand(batch_size, layer_size, -1)
        lut_indices = indices.unsqueeze(-1).expand(-1, -1, output_dim)
        
        output = luts[layer_indices[0], output_indices[0], lut_indices]  # (batch_size, layer_size, output_dim)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Probabilistic gradients like UnboundProbabilistic.
        
        This provides smooth gradients even though forward was discrete.
        """
        x, luts, binary_combinations = ctx.saved_tensors
        batch_size, layer_size, input_dim = x.shape
        output_dim = luts.size(1)
        lut_size = luts.size(2)
        
        # Input x is already in [0, 1] range, use directly for probabilistic computation
        x_prob = x  # (batch_size, layer_size, input_dim)
        
        # Compute probabilistic expectation for gradient
        # binary_combinations: (2^input_dim, input_dim)
        
        # Gradient w.r.t. inputs
        grad_input = torch.zeros_like(x)
        
        # Gradient w.r.t. LUTs
        grad_luts = torch.zeros_like(luts)
        
        # Process each layer independently
        for l in range(layer_size):
            x_l = x_prob[:, l, :]  # (batch_size, input_dim)
            
            # Expand for broadcasting
            x_expanded = x_l.unsqueeze(1)  # (batch_size, 1, input_dim)
            a_expanded = binary_combinations.unsqueeze(0)  # (1, 2^input_dim, input_dim)
            
            # Compute Pr(a|x) = prod_j [x_j^a_j * (1-x_j)^(1-a_j)]
            prob_terms = x_expanded * a_expanded + (1 - x_expanded) * (1 - a_expanded)
            probs = torch.prod(prob_terms, dim=2)  # (batch_size, 2^input_dim)
            
            # Gradient w.r.t. LUTs for this layer
            # grad_luts[l] = probs^T @ grad_output[:, l, :]
            grad_output_l = grad_output[:, l, :]  # (batch_size, output_dim)
            grad_luts[l] = torch.matmul(probs.t(), grad_output_l).t()  # (output_dim, 2^input_dim)
            
            # Gradient w.r.t. inputs for this layer
            for j in range(input_dim):
                # Derivative of probability w.r.t. x_j
                eps = 1e-8
                a_j = binary_combinations[:, j].unsqueeze(0)  # (1, 2^input_dim)
                x_j = x_l[:, j].unsqueeze(1)  # (batch_size, 1)
                
                # Compute derivative factor
                deriv_factor = (a_j - x_j) / (x_j * (1 - x_j) + eps)
                
                # Weight by probability
                prob_deriv = probs * deriv_factor  # (batch_size, 2^input_dim)
                
                # Sum over LUT entries weighted by LUT values and output gradient
                for dim in range(output_dim):
                    lut_weights = luts[l, dim, :].unsqueeze(0)  # (1, 2^input_dim)
                    grad_j = torch.sum(prob_deriv * lut_weights, dim=1)  # (batch_size,)
                    grad_input[:, l, j] += grad_j * grad_output_l[:, dim]
        
        return grad_input, grad_luts, None


def hybrid_forward(input, luts, binary_combinations):
    """
    Hybrid forward pass with automatic differentiation support.
    Forward: Binary thresholding at 0.5 (efficient, discrete)
    Backward: Probabilistic gradients (smooth, trainable)
    Updated for 3D tensors with per-layer-node parameters (no mapping - dense connectivity)
    
    Args:
        input: (batch_size, layer_size, input_dim) tensor in [0, 1]
        luts: (layer_size, output_dim, 2^input_dim) tensor in [0, 1] - per-layer-node LUTs
        binary_combinations: (2^input_dim, input_dim) tensor - precomputed binary patterns
    
    Returns:
        output: (batch_size, layer_size, output_dim) tensor
    """
    if _HYBRID_CUDA_EXT_AVAILABLE and input.is_cuda:
        return HybridFunction.apply(input, luts, binary_combinations)
    else:
        # CPU fallback
        return HybridFunctionCPU.apply(input, luts, binary_combinations)

@register_node("hybrid")
class HybridNode(BaseNode):
    """
    Hybrid LUT node combining DWN and UnboundProbabilistic approaches.
    
    Forward pass: Uses binary thresholding like DWN (discrete, efficient)
    Backward pass: Uses probabilistic gradients like UnboundProbabilistic (smooth, trainable)
    
    This combines the best of both worlds:
    - Fast, discrete inference (DWN)
    - Smooth, effective gradients (UnboundProbabilistic)
    
    Now supports per-layer-node LUTs for better memory access patterns.
    """
    
    def __init__(self, 
                 input_dim: int = None,
                 output_dim: int = None,
                 layer_size: int = None,
                 use_cuda: bool = True,
                 regularizers: dict = None,
                 init_fn: Optional[Callable] = None,
                 init_kwargs: dict = None):
        """
        Args:
            input_dim: Input dimensions (e.g., 6)
            output_dim: Output dimensions (e.g., 1)
            layer_size: Number of parallel nodes in the layer (e.g., 128)
            use_cuda: Whether to use CUDA kernels (if available)
            regularizers: Dict of custom regularization functions
            init_fn: Optional initialization function. Should take (param: torch.Tensor, **kwargs)
            init_kwargs: Keyword arguments for init_fn
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, layer_size=layer_size,
                         regularizers=regularizers, init_fn=init_fn, init_kwargs=init_kwargs)
        self.use_cuda = use_cuda and is_cuda_available()
        
        # Initialize raw LUT weights with per-layer-node LUTs
        # Shape: (layer_size, num_outputs, 2^num_inputs)
        lut_size = 2 ** self.num_inputs
        self.raw_luts = nn.Parameter(torch.randn(self.layer_size, self.num_outputs, lut_size) * 0.1)
        self._apply_init_fn(self.raw_luts, name="raw_luts")
        
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
        Returns: (layer_size, num_outputs, 2^num_inputs)
        """
        return torch.sigmoid(self.raw_luts)
         
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training.
        Inputs are in [0, 1], binarized using Heaviside at 0.5 (forward).
        Backward uses probabilistic gradients for smooth training.
        
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
        
        # Use hybrid forward (CUDA if available, else CPU)
        output = hybrid_forward(x, luts, self.binary_combinations)
        
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
        """No built-in regularization."""
        return torch.tensor(0.0, device=self.raw_luts.device, requires_grad=False)


# Export
__all__ = ['HybridNode']
