import torch
import torch.nn as nn
from typing import Optional, Callable
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

# Try to import the fused CUDA extension (for memory-efficient mapping)
try:
    import efd_fused_cuda as _efd_fused_cuda_module
    _FUSED_CUDA_EXT_AVAILABLE = True
except ImportError:
    _FUSED_CUDA_EXT_AVAILABLE = False
    _efd_fused_cuda_module = None
    # Don't warn - fused extension is optional optimization


class EFDFunction(torch.autograd.Function):
    """
    PyTorch autograd function wrapper for EFD CUDA kernels.
    Handles 3D tensors with per-layer-node parameters.
    """
    @staticmethod
    def forward(ctx, input, luts, alpha, beta):
        """
        Forward pass using CUDA kernel.
        
        Args:
            input: (batch_size, layer_size, input_dim) float tensor in [0, 1]
            luts: (layer_size, output_dim, 2^input_dim) float tensor in [0, 1]
            alpha: scalar float for gradient scaling
            beta: scalar float for Hamming distance decay
        
        Returns:
            output: (batch_size, layer_size, output_dim) float tensor in [0, 1]
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
    def backward(ctx, grad_output):
        """
        Backward pass using CUDA kernel with alpha/beta scaling.
        
        Args:
            grad_output: (batch_size, layer_size, output_dim) gradient tensor
        
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
    Handles 3D tensors with per-layer-node parameters.
    """
    @staticmethod
    def forward(ctx, input, luts, alpha, beta):
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
        ctx.alpha = alpha
        ctx.beta = beta
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using Extended Finite Difference (EFD) with alpha/beta scaling."""
        input, luts = ctx.saved_tensors
        alpha = ctx.alpha
        beta = ctx.beta
        
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
                    grad_luts[layer_idx, dim_idx, addr] += grad_output[batch_idx, layer_idx, dim_idx]
                
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
                            lut_value = luts[layer_idx, dim_idx, k].item()
                            
                            # Add weighted contribution: alpha * sign * lut * beta^hamming_dist
                            total_gradient += (alpha * sign_factor * lut_value * 
                                             (beta ** hamming_dist) * 
                                             grad_output[batch_idx, layer_idx, dim_idx].item())
                    
                    grad_input[batch_idx, layer_idx, input_idx] = total_gradient
        
        return grad_input, grad_luts, None, None


class EFDFusedFunction(torch.autograd.Function):
    """
    PyTorch autograd function wrapper for fused EFD CUDA kernels.
    Performs mapping and LUT lookup in a single kernel, avoiding materialization
    of (batch_size, layer_size, input_dim) intermediate tensor.
    """
    @staticmethod
    def forward(ctx, input, mapping_indices, luts, alpha, beta):
        """
        Fused forward pass using CUDA kernel.

        Args:
            input: (batch_size, input_size) float tensor in [0, 1]
            mapping_indices: (layer_size, input_dim) int64 tensor - indices into input_size
            luts: (layer_size, output_dim, 2^input_dim) float tensor in [0, 1]
            alpha: scalar float for gradient scaling
            beta: scalar float for Hamming distance decay

        Returns:
            output: (batch_size, layer_size, output_dim) float tensor in [0, 1]
        """
        if not _FUSED_CUDA_EXT_AVAILABLE:
            raise RuntimeError("Fused CUDA extension not available. Please compile with: python setup.py install")

        # Ensure correct dtypes and contiguity
        input = input.contiguous().float()
        mapping_indices = mapping_indices.contiguous().long()
        luts = luts.contiguous().float()

        # Call fused CUDA forward kernel
        output = _efd_fused_cuda_module.forward(input, mapping_indices, luts)

        # Save for backward
        ctx.save_for_backward(input, mapping_indices, luts)
        ctx.alpha = alpha
        ctx.beta = beta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using fused CUDA kernel with alpha/beta scaling.

        Args:
            grad_output: (batch_size, layer_size, output_dim) gradient tensor

        Returns:
            Gradients for (input, mapping_indices, luts, alpha, beta)
        """
        if not _FUSED_CUDA_EXT_AVAILABLE:
            raise RuntimeError("Fused CUDA extension not available. Please compile with: python setup.py install")

        input, mapping_indices, luts = ctx.saved_tensors
        alpha = ctx.alpha
        beta = ctx.beta

        # Ensure contiguity
        grad_output = grad_output.contiguous().float()

        # Call fused CUDA backward kernel with alpha and beta
        grad_input, grad_luts = _efd_fused_cuda_module.backward(
            input, mapping_indices, luts, grad_output, alpha, beta
        )

        # Return gradients (None for mapping_indices, alpha, beta)
        return grad_input, None, grad_luts, None, None


def efd_forward(input, luts, alpha, beta):
    """
    EFD forward pass with automatic differentiation support.
    
    Args:
        input: (batch_size, layer_size, input_dim) tensor in [0, 1]
        luts: (layer_size, output_dim, 2^input_dim) tensor in [0, 1]
        alpha: scalar float for gradient scaling
        beta: scalar float for Hamming distance decay
    
    Returns:
        output: (batch_size, layer_size, output_dim) tensor in [0, 1]
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
    Now supports per-layer-node LUTs for better memory access patterns.
    """
    
    def __init__(self, 
                 input_dim: int = None,
                 output_dim: int = None,
                 layer_size: int = None,
                 use_cuda: bool = True,
                 regularizers: dict = None,
                 alpha: float = None,
                 beta: float = None,
                 clamp_luts: bool = True,
                 init_fn: Optional[Callable] = None,
                 init_kwargs: dict = None):
        """
        Args:
            input_dim: Number of inputs (e.g., 6)
            output_dim: Number of outputs (e.g., 1)
            layer_size: Number of parallel nodes in the layer (e.g., 128)
            use_cuda: Whether to use CUDA kernels (if available)
            regularizers: Dict of custom regularization functions
            alpha: Gradient scaling factor (default: 0.5 * 0.75^(n-1))
            beta: Hamming distance decay factor (default: 0.25/0.75)
            clamp_luts: Whether to clamp LUT values to [0, 1] during training
            init_fn: Optional initialization function for LUT weights. Should take (param: torch.Tensor, **kwargs)
            init_kwargs: Keyword arguments for init_fn
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, layer_size=layer_size,
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
            alpha = 0.5 * (0.75 ** (self.num_inputs - 1))
        if beta is None:
            beta = 0.25 / 0.75
        
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer('beta', torch.tensor(beta, dtype=torch.float32))
        
        # Initialize LUT weights with per-layer-node LUTs
        # Shape: (layer_size, num_outputs, 2^num_inputs)
        lut_size = 2 ** self.num_inputs
        self.luts = nn.Parameter(torch.rand(self.layer_size, self.num_outputs, lut_size))
        self._apply_init_fn(self.luts, name="luts")
         
    def _clamp_luts_if_needed(self):
        """Clamp LUT values to [0, 1] during training if enabled."""
        if self.training and self.clamp_luts:
            with torch.no_grad():
                self.luts.clamp_(0.0, 1.0)
    
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training.
        Inputs are in [0, 1], binarized using Heaviside at 0.5.
        Outputs are in [0, 1].
        Uses CUDA kernels when available, otherwise CPU fallback.
        
        Args:
            x: Tensor of shape (batch_size, layer_size, input_dim)
        
        Returns:
            Tensor of shape (batch_size, layer_size, output_dim)
        """
        # Clamp LUT values to [0, 1] if enabled
        self._clamp_luts_if_needed()
        
        batch_size, layer_size, input_dim = x.shape
        
        # Verify layer_size matches
        if layer_size != self.layer_size:
            raise ValueError(
                f"Input layer_size {layer_size} does not match node's layer_size {self.layer_size}"
            )
        
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
            x: Tensor of shape (batch_size, layer_size, input_dim)
        
        Returns:
            Tensor of shape (batch_size, layer_size, output_dim)
        """
        batch_size, layer_size, input_dim = x.shape
        
        # Verify layer_size matches
        if layer_size != self.layer_size:
            raise ValueError(
                f"Input layer_size {layer_size} does not match node's layer_size {self.layer_size}"
            )
        
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
        
        output = self.luts[layer_indices, output_indices, lut_indices]  # (batch_size, layer_size, num_outputs)
        
        # Binarize output: [0, 1] -> {0, 1} using threshold at 0.5
        output = (output >= 0.5).float()
        
        return output

    def forward_with_mapping(self, x: torch.Tensor, mapping_indices: torch.Tensor) -> torch.Tensor:
        """
        Fused forward pass: mapping + LUT lookup in single CUDA kernel.
        Avoids materializing mapped_inputs tensor, saving massive memory.

        This method is called by BaseLUTLayer when get_mapping_indices() is available.
        It performs on-the-fly indexing inside the CUDA kernel instead of pre-materializing
        the (batch_size, layer_size, input_dim) intermediate tensor.

        Args:
            x: Input tensor (batch_size, input_size)
               Raw 2D input from encoder or previous layer
            mapping_indices: Indices (layer_size, num_inputs) int64 tensor
                           Values in range [0, input_size), defines which inputs to select

        Returns:
            Output tensor (batch_size, layer_size, num_outputs)

        Memory Impact:
            - Without fusion: Creates (batch, layer_size, input_dim) tensor = ~60 MB per layer
            - With fusion: Only uses tiny mapping buffer = ~2 KB
            - Savings: 99.997% memory reduction for mapping overhead
        """
        if self.training:
            self._clamp_luts_if_needed()

        batch_size = x.shape[0]
        input_size = x.shape[1]

        # Validate dimensions
        if mapping_indices.shape != (self.layer_size, self.num_inputs):
            raise ValueError(
                f"Expected mapping shape ({self.layer_size}, {self.num_inputs}), "
                f"got {mapping_indices.shape}"
            )

        # Use fused CUDA kernel if available
        if self.use_cuda and x.is_cuda and _FUSED_CUDA_EXT_AVAILABLE:
            # Fused path: no intermediate tensor materialized
            output = EFDFusedFunction.apply(
                x, mapping_indices, self.luts,
                self.alpha.item(), self.beta.item()
            )
        else:
            # Fallback to base implementation (materializes mapped_inputs)
            # This happens when:
            # - CUDA not available
            # - Fused extension not compiled
            # - CPU tensors
            output = super().forward_with_mapping(x, mapping_indices)

        return output

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization to match base CUDA implementation."""
        return torch.tensor(0.0, device=self.luts.device, requires_grad=False)