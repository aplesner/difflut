import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Optional, Tuple
import warnings
from .base_layer import BaseLUTLayer
from ..registry import register_layer
from ..nodes.node_config import NodeConfig
from ..utils.warnings import warn_default_value

# Default temperature for softmax in learnable mapping
DEFAULT_LEARNABLE_LAYER_TAU: float = 0.001
# Default starting value for tau (used for exponential decay)
DEFAULT_LEARNABLE_LAYER_TAU_START: float = 1.0
# Default minimum value tau can decay to
DEFAULT_LEARNABLE_LAYER_TAU_MIN: float = 0.0001
# Default number of iterations for tau to decay by factor of 10
DEFAULT_LEARNABLE_LAYER_TAU_DECAY_ITERS: float = 1000.0
# Threshold for warning about excessive learnable connections
# If output_size * n > input_size * LEARNABLE_LAYER_CONNECTION_WARNING_THRESHOLD, warn
LEARNABLE_LAYER_CONNECTION_WARNING_THRESHOLD: int = 10
# Use CUDA kernel for soft selection (training mode)
# If False, always use PyTorch's matmul (more stable, well-tested)
# If True, use custom CUDA kernel (potentially faster for large matrices)
# Note: Hard selection (eval mode) always uses CUDA kernel when available
DEFAULT_LEARNABLE_LAYER_USE_CUDA_SOFT: bool = False

# Try to import the compiled CUDA extension for learnable mapping
try:
    import learnable_mapping_cuda as _learnable_mapping_cuda_module
    _LEARNABLE_MAPPING_CUDA_AVAILABLE = True
except ImportError:
    _LEARNABLE_MAPPING_CUDA_AVAILABLE = False
    _learnable_mapping_cuda_module = None
    warnings.warn(
        "CUDA extension 'learnable_mapping_cuda' not available. LearnableLayer will use PyTorch fallback. "
        "For better performance, compile the CUDA extension using: "
        "'cd difflut && python setup.py install'. "
        "To suppress this warning: warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut.layers.learnable_layer')",
        RuntimeWarning,
        stacklevel=2
    )


class LearnableMappingFunction(torch.autograd.Function):
    """
    Autograd function wrapper for learnable mapping CUDA kernel (hard selection).
    """
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, input: torch.Tensor, indices: torch.Tensor, input_size: int) -> torch.Tensor:
        """
        Forward pass using CUDA kernel for hard selection.
        
        Args:
            input: (batch_size, input_size) float tensor
            indices: (output_size,) int32 tensor - argmax results
            input_size: int (needed for backward)
        
        Returns:
            output: (batch_size, output_size) float tensor
        """
        if not _LEARNABLE_MAPPING_CUDA_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Use fallback implementation.")
        
        # Ensure correct dtypes and contiguity
        input = input.contiguous().float()
        indices = indices.contiguous().int()
        
        # Call CUDA forward kernel
        output = _learnable_mapping_cuda_module.forward(input, indices)
        
        # Save for backward
        ctx.save_for_backward(indices)
        ctx.input_size = input_size
        
        return output
    
    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """
        Backward pass using CUDA kernel.
        
        Args:
            grad_output: (batch_size, output_size) gradient tensor
        
        Returns:
            Gradients for (input, indices, input_size)
        """
        if not _LEARNABLE_MAPPING_CUDA_AVAILABLE:
            raise RuntimeError("CUDA extension not available.")
        
        indices, = ctx.saved_tensors
        input_size = ctx.input_size
        
        # Ensure contiguity
        grad_output = grad_output.contiguous().float()
        
        # Call CUDA backward kernel
        grad_input = _learnable_mapping_cuda_module.backward(grad_output, indices, input_size)
        
        # Return gradients (None for indices and input_size)
        return grad_input, None, None


def learnable_mapping_forward_cuda(input: torch.Tensor, indices: torch.Tensor, input_size: int) -> Optional[torch.Tensor]:
    """
    Learnable mapping forward pass (hard selection) with CUDA.
    
    Args:
        input: (batch_size, input_size) tensor
        indices: (output_size,) tensor
        input_size: int
    
    Returns:
        output: (batch_size, output_size) tensor
    """
    if _LEARNABLE_MAPPING_CUDA_AVAILABLE and input.is_cuda:
        return LearnableMappingFunction.apply(input, indices, input_size)
    else:
        return None


def learnable_mapping_soft_forward_cuda(input: torch.Tensor, weights: torch.Tensor, tau: float) -> Optional[torch.Tensor]:
    """
    Learnable mapping forward pass (soft selection) with CUDA.
    
    Args:
        input: (batch_size, input_size) tensor
        weights: (output_size, input_size) tensor
        tau: float
    
    Returns:
        output: (batch_size, output_size) tensor
    """
    if _LEARNABLE_MAPPING_CUDA_AVAILABLE and input.is_cuda:
        try:
            return _learnable_mapping_cuda_module.soft_forward(input, weights, tau)
        except:
            return None
    else:
        return None


class LearnableMappingModule(nn.Module):
    """
    Helper module for learnable mapping (not registered, used internally).
    Provides soft selection during training and hard selection during evaluation.
    Uses CUDA kernels when available for optimal performance.
    
    Note: output_size here is actually (layer_output_size * n) - the total number of
    selections needed. This module doesn't know about the layer structure.
    """
    
    def __init__(self, input_size: int, output_size: int, tau: float = DEFAULT_LEARNABLE_LAYER_TAU, 
                 use_cuda_soft: bool = DEFAULT_LEARNABLE_LAYER_USE_CUDA_SOFT):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size  # This is actually layer_output_size * n
        self.tau = tau
        self.use_cuda_soft = use_cuda_soft
        
        # Weight matrix: (output_size, input_size) where output_size = layer_output_size * n
        self.W = nn.Parameter(torch.randn(output_size, input_size))
        nn.init.xavier_uniform_(self.W)
        
        # Cache for hard selection (indices instead of mask for CUDA kernel)
        self.register_buffer('_cached_hard_indices', None)
        self.register_buffer('_cached_hard_mask', None)  # Keep for PyTorch fallback
        self._cache_valid = False
    
    def train(self, mode: bool = True):
        """Override train() to invalidate cache when switching modes."""
        was_training = self.training
        super().train(mode)
        
        # Invalidate cache when switching from eval to train
        if mode and not was_training:
            self._cache_valid = False
        
        return self
    
    def _compute_hard_selection(self) -> None:
        """
        Compute hard selection from current weights.
        Computes both indices (for CUDA kernel) and mask (for PyTorch fallback).
        """
        with torch.no_grad():
            # Get hard indices: which input is selected for each output position
            # W shape: (output_size, input_size) where output_size = layer_output_size * n
            hard_indices = torch.argmax(self.W, dim=-1)  # (output_size,)
            
            # Store indices for CUDA kernel
            self._cached_hard_indices = hard_indices.int()
            
            # Also create binary mask for PyTorch fallback
            # mask: (input_size, output_size) where mask[i, o] = 1 if input i is selected for output o
            mask = torch.zeros((self.input_size, self.output_size), dtype=torch.uint8, device=self.W.device)
            output_indices = torch.arange(self.output_size, device=self.W.device)
            mask[hard_indices, output_indices] = 1
            self._cached_hard_mask = mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Soft selection (training) or hard selection (eval).
        Uses CUDA kernels when available for optimal performance.
        Training: softmax + matmul (PyTorch default, or optional CUDA kernel)
        Eval: CUDA kernel for direct lookup (faster than einsum)
        """
        if self.training:
            # Soft selection - training mode
            # Try CUDA kernel first if enabled (can be faster for very large matrices)
            if self.use_cuda_soft and _LEARNABLE_MAPPING_CUDA_AVAILABLE and x.is_cuda:
                output = learnable_mapping_soft_forward_cuda(x, self.W, self.tau)
                if output is not None:
                    return output
            
            # PyTorch fallback - already well optimized
            weights = F.softmax(self.W / self.tau, dim=-1)
            output = torch.matmul(x, weights.t())
        else:
            # Hard selection (evaluation mode)
            # OPTIMIZATION: Cache hard selection to avoid repeated argmax computation
            if not self._cache_valid or self._cached_hard_indices is None:
                self._compute_hard_selection()
                self._cache_valid = True
            
            # Try CUDA kernel first (fastest)
            if _LEARNABLE_MAPPING_CUDA_AVAILABLE and x.is_cuda:
                output = learnable_mapping_forward_cuda(x, self._cached_hard_indices, self.input_size)
                if output is not None:
                    return output
            
            # PyTorch fallback - use einsum with binary mask
            # x: (batch_size, input_size) -> (b, i)
            # _cached_hard_mask: (input_size, output_size) -> (i, o)
            # Result: (batch_size, output_size) -> (b, o)
            mask_float = self._cached_hard_mask.float()
            output = torch.einsum('bi,io->bo', x, mask_float)
        
        return output


@register_layer("learnable")
class LearnableLayer(BaseLUTLayer):
    """
    LUT layer with learnable mapping using nodes.
    Uses soft selection during training and hard selection during evaluation.
    
    The learnable mapping uses a weight matrix W and applies softmax for soft
    selection during training, or argmax for hard selection during evaluation.
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int, 
                 node_type: Type[nn.Module],
                 node_kwargs: NodeConfig,
                 tau: float = None,
                 tau_start: float = None,
                 tau_min: float = None,
                 tau_decay_iters: float = None,
                 flip_probability: float = None,
                 grad_stabilization: str = None,
                 grad_target_std: float = None,
                 grad_subtract_mean: bool = None,
                 grad_epsilon: float = None,
                 max_nodes_per_batch: int = None,
                 use_cuda_soft: bool = None):
        """
        Args:
            input_size: Size of input vector (from encoder or previous layer)
                       Should match: (batch_size, input_size)
            output_size: Number of LUT nodes (output will be batch_size, output_size * output_dim)
            node_type: LUT node class
            node_kwargs: Node configuration (NodeConfig instance with input_dim, output_dim, etc.)
                        Dimension spec: nodes expect (batch_size, output_size, node_input_dim)
            tau: Initial temperature for softmax in learnable mapping
            tau_start: Starting value for tau (used for exponential decay)
            tau_min: Minimum value tau can decay to
            tau_decay_iters: Number of iterations for tau to decay by factor of 10
            flip_probability: Probability of flipping each bit during training (0.0 to 1.0)
            grad_stabilization: Gradient stabilization mode ('none', 'layerwise', 'batchwise')
            grad_target_std: Target standard deviation for gradient rescaling
            grad_subtract_mean: Whether to subtract mean before rescaling
            grad_epsilon: Small constant for numerical stability
            max_nodes_per_batch: Maximum nodes to process per batch (memory optimization)
            use_cuda_soft: Use CUDA kernel for soft selection (training mode)
        """
        # Set defaults from constants
        if tau is None:
            tau = DEFAULT_LEARNABLE_LAYER_TAU
            warn_default_value("tau", tau, stacklevel=2)
        if tau_start is None:
            tau_start = DEFAULT_LEARNABLE_LAYER_TAU_START
            warn_default_value("tau_start", tau_start, stacklevel=2)
        if tau_min is None:
            tau_min = DEFAULT_LEARNABLE_LAYER_TAU_MIN
            warn_default_value("tau_min", tau_min, stacklevel=2)
        if tau_decay_iters is None:
            tau_decay_iters = DEFAULT_LEARNABLE_LAYER_TAU_DECAY_ITERS
            warn_default_value("tau_decay_iters", tau_decay_iters, stacklevel=2)
        if use_cuda_soft is None:
            use_cuda_soft = DEFAULT_LEARNABLE_LAYER_USE_CUDA_SOFT
            warn_default_value("use_cuda_soft", use_cuda_soft, stacklevel=2)
        
        # Warn if tau parameters seem unusual
        if tau_start < tau_min:
            warnings.warn(
                f"LearnableLayer: tau_start ({tau_start}) is less than tau_min ({tau_min}). "
                f"This means tau will be clamped immediately. Set tau_start >= tau_min.",
                UserWarning,
                stacklevel=2
            )
        
        # Initialize parent with nodes (n will be extracted from created nodes)
        super().__init__(input_size, output_size, node_type, node_kwargs, flip_probability,
                        grad_stabilization, grad_target_std, grad_subtract_mean, grad_epsilon,
                        max_nodes_per_batch)
        
        # Warn about parameter count after n is known
        total_connections = output_size * self.n
        if total_connections > input_size * LEARNABLE_LAYER_CONNECTION_WARNING_THRESHOLD:
            warnings.warn(
                f"LearnableLayer: Creating {total_connections} learnable connections from {input_size} inputs. "
                f"This may lead to overfitting. Consider using GroupedLayer or fewer nodes/inputs per node (n={self.n}).",
                UserWarning,
                stacklevel=2
            )
        
        # Tau decay parameters
        self.tau_start = tau_start
        self.tau_min = tau_min
        self.tau_decay_iters = tau_decay_iters
        self.tau = tau_start  # Start with tau_start instead of tau
        self.use_cuda_soft = use_cuda_soft
        
        # Create learnable mapping module (helper, not registered)
        # Note: self.n is now available from parent's __init__
        self.mapping = LearnableMappingModule(input_size, output_size * self.n, self.tau, use_cuda_soft)
    
    def get_mapping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable mapping and reshape for nodes.
        
        Uses efficient matrix operations (training) or advanced indexing (eval).
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        Returns:
            Mapped inputs of shape (batch_size, output_size, n)
        """
        batch_size = x.shape[0]
        
        # Apply learnable mapping (already optimized in LearnableMappingModule)
        # Training: uses matmul (efficient)
        # Eval: uses advanced indexing (efficient)
        mapped_flat = self.mapping(x)  # (batch_size, output_size * n)
        
        # Reshape for nodes
        mapped_inputs = mapped_flat.view(batch_size, self.output_size, self.n)
        
        return mapped_inputs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through learnable mapping and node.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output tensor of shape (batch_size, output_size * output_dim)
        """
        # Validate input dimensions
        self._validate_input_dims(x)
        
        # Get mapped inputs: (batch_size, output_size, n)
        mapped_inputs = self.get_mapping(x)
        
        # MEMORY OPTIMIZATION: Process nodes in batches if output_size is large
        # This prevents OOM errors by chunking the layer dimension
        batch_size = mapped_inputs.shape[0]
        
        if self.max_nodes_per_batch > 0 and self.output_size > self.max_nodes_per_batch:
            # Process nodes in chunks to reduce memory usage
            output = self._forward_with_node_batching(mapped_inputs)
        else:
            # Standard path: process all nodes at once
            output = self.node(mapped_inputs)
        
        # Output shape: (batch_size, output_size, output_dim)
        # Reshape to 2D for next layer: (batch_size, output_size * output_dim)
        output = output.view(batch_size, -1)
        
        return output
    
    def _forward_with_node_batching(self, mapped_inputs: torch.Tensor) -> torch.Tensor:
        """
        Process nodes in batches to reduce memory usage.
        
        Splits the layer dimension into chunks and processes each chunk separately,
        slicing both inputs and node parameters appropriately.
        
        Args:
            mapped_inputs: (batch_size, output_size, n) tensor
        
        Returns:
            output: (batch_size, output_size, output_dim) tensor
        """
        batch_size, output_size, n = mapped_inputs.shape
        
        # Preallocate output tensor
        output = torch.empty(
            (batch_size, output_size, self.node.output_dim),
            device=mapped_inputs.device,
            dtype=mapped_inputs.dtype
        )
        
        # Process nodes in chunks
        for start_idx in range(0, output_size, self.max_nodes_per_batch):
            end_idx = min(start_idx + self.max_nodes_per_batch, output_size)
            
            # Extract chunk: (batch_size, chunk_size, n)
            mapped_chunk = mapped_inputs[:, start_idx:end_idx, :]
            
            # Process chunk through node with parameter slicing
            # The node should handle slicing its own parameters based on layer indices
            output_chunk = self._process_node_chunk(mapped_chunk, start_idx, end_idx)
            
            # Store result
            output[:, start_idx:end_idx, :] = output_chunk
        
        return output
    
    def _process_node_chunk(self, mapped_chunk: torch.Tensor, start_idx: int, end_idx: int) -> torch.Tensor:
        """
        Process a chunk of nodes by slicing node parameters.
        
        Args:
            mapped_chunk: (batch_size, chunk_size, n) tensor
            start_idx: Start index in the layer
            end_idx: End index in the layer
        
        Returns:
            output_chunk: (batch_size, chunk_size, output_dim) tensor
        """
        from ..nodes.probabilistic_node import ProbabilisticNode
        from ..nodes.linear_lut_node import LinearLUTNode
        
        # Check node type and slice parameters accordingly
        if isinstance(self.node, ProbabilisticNode):
            # Slice raw_weights: (layer_size, 2^n, output_dim) -> (chunk_size, 2^n, output_dim)
            raw_weights_chunk = self.node.raw_weights[start_idx:end_idx]
            
            # Call forward_train directly with sliced parameters
            if self.training:
                output_chunk = self._probabilistic_forward_chunk(
                    mapped_chunk, 
                    raw_weights_chunk,
                    self.node.temperature
                )
            else:
                # Evaluation mode
                output_chunk = self._probabilistic_eval_chunk(
                    mapped_chunk,
                    raw_weights_chunk,
                    self.node.temperature
                )
        elif isinstance(self.node, LinearLUTNode):
            # Slice weights: (layer_size, 2^n, output_dim) -> (chunk_size, 2^n, output_dim)
            weights_chunk = self.node.weights[start_idx:end_idx]
            output_chunk = self._linear_lut_forward_chunk(mapped_chunk, weights_chunk)
        else:
            # For other node types, try generic parameter slicing
            # This may not work for all node types - they may need custom handling
            output_chunk = self._generic_forward_chunk(mapped_chunk, start_idx, end_idx)
        
        return output_chunk
    
    def _probabilistic_forward_chunk(
        self, 
        x: torch.Tensor, 
        raw_weights: torch.Tensor,
        temperature: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for a chunk of ProbabilisticNodes.
        
        Args:
            x: (batch_size, chunk_size, input_dim) tensor
            raw_weights: (chunk_size, 2^input_dim, output_dim) tensor
            temperature: scalar tensor
        
        Returns:
            output: (batch_size, chunk_size, output_dim) tensor
        """
        # Try CUDA kernel first
        if self.node.use_cuda and x.is_cuda:
            try:
                import probabilistic_cuda
                output = probabilistic_cuda.forward(x.contiguous(), raw_weights.contiguous(), float(temperature))
                return output
            except:
                pass
        
        # CPU fallback
        batch_size, chunk_size, input_dim = x.shape
        weights = torch.sigmoid(raw_weights / temperature.clamp(min=1e-6))
        
        # Ensure binary_combinations is on the same device
        binary_combinations = self.node.binary_combinations.to(device=x.device, dtype=x.dtype)
        
        # Preallocate output
        output = torch.empty(
            (batch_size, chunk_size, self.node.output_dim),
            device=x.device,
            dtype=x.dtype
        )
        
        # Process each layer in chunk
        for layer_idx in range(chunk_size):
            x_layer = x[:, layer_idx, :]  # (batch_size, input_dim)
            
            # Vectorized probability computation
            x_expanded = x_layer.unsqueeze(1)  # (batch_size, 1, input_dim)
            a_expanded = binary_combinations.unsqueeze(0)  # (1, 2^input_dim, input_dim)
            prob_terms = x_expanded * a_expanded + (1 - x_expanded) * (1 - a_expanded)
            probs = torch.prod(prob_terms, dim=-1)  # (batch_size, 2^input_dim)
            
            # Apply weights for this layer
            output[:, layer_idx, :] = torch.matmul(probs, weights[layer_idx])
        
        return output
    
    def _probabilistic_eval_chunk(
        self,
        x: torch.Tensor,
        raw_weights: torch.Tensor,
        temperature: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluation forward pass for a chunk of ProbabilisticNodes.
        
        Args:
            x: (batch_size, chunk_size, input_dim) tensor
            raw_weights: (chunk_size, 2^input_dim, output_dim) tensor
            temperature: scalar tensor
        
        Returns:
            output: (batch_size, chunk_size, output_dim) tensor
        """
        batch_size, chunk_size, input_dim = x.shape
        weights = torch.sigmoid(raw_weights / temperature.clamp(min=1e-6))
        
        # Convert binary inputs to LUT indices
        powers = 2 ** torch.arange(input_dim, device=x.device, dtype=x.dtype)
        indices = (x * powers).sum(dim=-1).long()  # (batch_size, chunk_size)
        
        # Preallocate output
        output = torch.empty(
            (batch_size, chunk_size, self.node.output_dim),
            device=x.device,
            dtype=weights.dtype
        )
        
        # Gather per-layer weights
        for layer_idx in range(chunk_size):
            batch_indices = indices[:, layer_idx]  # (batch_size,)
            output[:, layer_idx, :] = weights[layer_idx][batch_indices]  # (batch_size, output_dim)
        
        # Threshold to get binary output
        output = (output >= 0.5).float()
        
        return output
    
    def _linear_lut_forward_chunk(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a chunk of LinearLUTNodes.
        
        Args:
            x: (batch_size, chunk_size, input_dim) tensor
            weights: (chunk_size, 2^input_dim, output_dim) tensor
        
        Returns:
            output: (batch_size, chunk_size, output_dim) tensor
        """
        batch_size, chunk_size, input_dim = x.shape
        
        # Convert binary inputs to LUT indices
        powers = 2 ** torch.arange(input_dim, device=x.device, dtype=x.dtype)
        indices = (x * powers).sum(dim=-1).long()  # (batch_size, chunk_size)
        
        # Preallocate output
        output = torch.empty(
            (batch_size, chunk_size, self.node.output_dim),
            device=x.device,
            dtype=weights.dtype
        )
        
        # Gather per-layer weights
        for layer_idx in range(chunk_size):
            batch_indices = indices[:, layer_idx]  # (batch_size,)
            output[:, layer_idx, :] = weights[layer_idx][batch_indices]  # (batch_size, output_dim)
        
        return output
    
    def _generic_forward_chunk(self, mapped_chunk: torch.Tensor, start_idx: int, end_idx: int) -> torch.Tensor:
        """
        Generic fallback for nodes that don't have custom chunking logic.
        Simply calls the node's forward method - may not actually save memory.
        
        Args:
            mapped_chunk: (batch_size, chunk_size, n) tensor
            start_idx: Start index in the layer
            end_idx: End index in the layer
        
        Returns:
            output_chunk: (batch_size, chunk_size, output_dim) tensor
        """
        # Warning: This may not actually reduce memory for all node types
        # Node types should implement their own chunking logic
        return self.node(mapped_chunk)
    
    def get_mapping_matrix(self) -> torch.Tensor:
        """Get current hard mapping (for inspection) as indices."""
        with torch.no_grad():
            # Get hard indices from weight matrix
            hard_indices = torch.argmax(self.mapping.W, dim=-1)  # (output_size * n,)
            return hard_indices.view(self.output_size, self.n)
    
    def update_tau(self, iteration: int):
        """
        Update tau using exponential decay.
        
        Args:
            iteration: Current training iteration
        """
        # Calculate decay factor: tau = tau_start * 10^(-iteration / tau_decay_iters)
        # This means tau decays by a factor of 10 every tau_decay_iters iterations
        decay_factor = 10.0 ** (-iteration / self.tau_decay_iters)
        self.tau = max(self.tau_start * decay_factor, self.tau_min)
        
        # Update the mapping module's tau
        self.mapping.tau = self.tau
        
        # Invalidate cache since tau change might affect hard selection
        # (though in practice, argmax is deterministic regardless of tau)
        self.mapping._cache_valid = False

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        flip_str = f", flip_prob={self.flip_probability}" if self.flip_probability > 0 else ""
        grad_str = f", grad_stab={self.grad_stabilization}" if self.grad_stabilization != 'none' else ""
        return f"input_size={self.input_size}, output_size={self.output_size}, " \
               f"n={self.n}, tau={self.tau}, mapping=learnable{flip_str}{grad_str}"