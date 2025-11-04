import torch
import torch.nn as nn
from typing import Type, Optional, Dict, Any, Tuple, Callable, List
import warnings
from .base_layer import BaseLUTLayer
from ..registry import register_layer
from ..nodes.node_config import NodeConfig
from ..utils.warnings import warn_default_value

# Default random seed for reproducible random mapping
DEFAULT_RANDOM_LAYER_SEED: int = 42


# Try to import the compiled CUDA extension for mapping
try:
    import mapping_cuda as _mapping_cuda_module
    _MAPPING_CUDA_AVAILABLE = True
except ImportError:
    _MAPPING_CUDA_AVAILABLE = False
    _mapping_cuda_module = None
    warnings.warn(
        "CUDA extension 'mapping_cuda' not available. RandomLayer will use slower PyTorch fallback. "
        "For better performance, compile the CUDA extension using: "
        "'cd difflut && python setup.py install'. "
        "To suppress this warning: warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut.layers.random_layer')",
        RuntimeWarning,
        stacklevel=2
    )

# Try to import the compiled CUDA extension for probabilistic nodes
try:
    import probabilistic_cuda as _probabilistic_cuda_module
    _PROBABILISTIC_CUDA_AVAILABLE = True
except ImportError:
    _PROBABILISTIC_CUDA_AVAILABLE = False
    _probabilistic_cuda_module = None


class MappingFunction(torch.autograd.Function):
    """
    Autograd function wrapper for mapping CUDA kernel.
    Provides forward and backward passes with automatic differentiation.
    """
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, input: torch.Tensor, indices: torch.Tensor, input_size: int) -> torch.Tensor:
        """
        Forward pass using CUDA kernel.
        
        Args:
            input: (batch_size, input_size) float tensor
            indices: (output_size, n) int16/int32 tensor
            input_size: int (needed for backward)
        
        Returns:
            output: (batch_size, output_size, n) float tensor
        """
        if not _MAPPING_CUDA_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Use fallback implementation.")
        
        # Ensure correct dtypes and contiguity
        input = input.contiguous().float()
        indices = indices.contiguous()
        
        # Call CUDA forward kernel
        output = _mapping_cuda_module.forward(input, indices)
        
        # Save for backward
        ctx.save_for_backward(indices)
        ctx.input_size = input_size
        
        return output
    
    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """
        Backward pass using CUDA kernel.
        
        Args:
            grad_output: (batch_size, output_size, n) gradient tensor
        
        Returns:
            Gradients for (input, indices, input_size)
        """
        if not _MAPPING_CUDA_AVAILABLE:
            raise RuntimeError("CUDA extension not available.")
        
        indices, = ctx.saved_tensors
        input_size = ctx.input_size
        
        # Ensure contiguity
        grad_output = grad_output.contiguous().float()
        
        # Call CUDA backward kernel
        grad_input = _mapping_cuda_module.backward(grad_output, indices, input_size)
        
        # Return gradients (None for indices and input_size as they don't need gradients)
        return grad_input, None, None


def mapping_forward_cuda(input: torch.Tensor, indices: torch.Tensor, input_size: int) -> Optional[torch.Tensor]:
    """
    Mapping forward pass with automatic differentiation support.
    
    Args:
        input: (batch_size, input_size) tensor
        indices: (output_size, n) tensor
        input_size: int
    
    Returns:
        output: (batch_size, output_size, n) tensor
    """
    if _MAPPING_CUDA_AVAILABLE and input.is_cuda:
        return MappingFunction.apply(input, indices, input_size)
    else:
        return None


@register_layer("random")
class RandomLayer(BaseLUTLayer):
    """
    LUT layer with purely random fixed mapping.
    Each input is used at least once per node before being reused.
    Connections are randomly initialized and remain fixed during training.
    
    Uses efficient index_select operations to minimize memory during mapping,
    avoiding large intermediate tensors.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        node_type: Type[nn.Module],
        node_kwargs: NodeConfig,
        seed: Optional[int] = DEFAULT_RANDOM_LAYER_SEED,
        flip_probability: Optional[float] = None,
        grad_stabilization: Optional[str] = None,
        grad_target_std: Optional[float] = None,
        grad_subtract_mean: Optional[bool] = None,
        grad_epsilon: Optional[float] = None,
        max_nodes_per_batch: Optional[int] = None
    ) -> None:
        """
        Args:
            input_size: Size of input vector (from encoder or previous layer)
                       Should match: (batch_size, input_size)
            output_size: Number of LUT nodes (output will be batch_size, output_size * output_dim)
            node_type: LUT node class to use
            node_kwargs: Node configuration (NodeConfig instance with input_dim, output_dim, etc.)
                        Dimension spec: nodes expect (batch_size, output_size, node_input_dim)
            seed: Random seed for reproducible mapping
            flip_probability: Probability of flipping each bit during training (0.0 to 1.0)
            grad_stabilization: Gradient stabilization mode ('none', 'layerwise', 'batchwise')
            grad_target_std: Target standard deviation for gradient rescaling
            grad_subtract_mean: Whether to subtract mean before rescaling
            grad_epsilon: Small constant for numerical stability
            max_nodes_per_batch: Maximum nodes to process per batch (memory optimization)
        """
        self.seed = seed
        
        # Note: Seed warning removed since seed is now explicitly provided in configs.
        # Only warn if seed is truly missing (None), not if it equals the default value.
        
        # Initialize parent (n will be extracted from created nodes)
        super().__init__(input_size, output_size, node_type, node_kwargs, flip_probability,
                        grad_stabilization, grad_target_std, grad_subtract_mean, grad_epsilon,
                        max_nodes_per_batch)
        
        # Initialize the random mapping
        self._init_mapping()
    
    def _init_mapping(self) -> None:
        """
        Initialize random mapping matrix.
        Ensures each input is used at least once per node before any reuse.
        
        Creates an index tensor of shape (output_size, n) where each entry specifies
        which input index to use. This is more memory efficient than the binary mask.
        """
        # Store current RNG state and set seed for reproducibility
        rng_state = torch.get_rng_state()
        torch.manual_seed(self.seed)
        
        # Create index mapping: (output_size, n)
        # For each output node and position, store which input index to use
        # Use int16 for memory efficiency (supports up to 32k input features)
        dtype = torch.int16 if self.input_size < 32768 else torch.int32
        mapping_indices = torch.zeros((self.output_size, self.n), dtype=dtype)

        for node_idx in range(self.output_size):
            # Calculate how many full cycles we need
            full_cycles = self.n // self.input_size
            remainder = self.n % self.input_size
            
            indices = []
            
            # For each full cycle, use all inputs once in random order
            for _ in range(full_cycles):
                perm = torch.randperm(self.input_size)
                indices.extend(perm.tolist())
            
            # For remainder, use a random subset of inputs
            if remainder > 0:
                perm = torch.randperm(self.input_size)
                indices.extend(perm[:remainder].tolist())
            
            # Store indices for this node
            mapping_indices[node_idx] = torch.tensor(indices, dtype=dtype)
        
        # Register as buffer (not a parameter, but saved with model)
        # Shape: (output_size, n) - index mapping (int16/int32)
        # Memory: output_size * n * 2 bytes (vs input_size * output_size * n * 1 byte for mask)
        # For typical case: 1000 * 6 * 2 = 12KB (vs 1568 * 1000 * 6 * 1 = 9.4MB)
        # Reduction: ~99% memory savings for mapping storage!
        self.register_buffer('_mapping_indices', mapping_indices)
        
        # Restore original RNG state
        torch.set_rng_state(rng_state)

    def get_mapping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get mapped inputs using the fixed random mapping.
        
        Uses custom CUDA kernel when available for optimal performance.
        Falls back to PyTorch gather operations on CPU or if CUDA extension unavailable.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Mapped inputs of shape (batch_size, output_size, n)
        """
        # Try CUDA kernel first (fastest, eliminates expand + gather overhead)
        if _MAPPING_CUDA_AVAILABLE and x.is_cuda:
            mapped_inputs = mapping_forward_cuda(x, self._mapping_indices, self.input_size)
            if mapped_inputs is not None:
                return mapped_inputs
        
        # PyTorch fallback - efficient gather-based implementation
        # MEMORY OPTIMIZATION: Use gather with index tensor
        # This is more memory efficient than einsum as it:
        # 1. Doesn't require float conversion of mask
        # 2. Uses optimized indexing kernels
        # 3. Avoids intermediate sparse matmul operations
        
        batch_size = x.shape[0]
        
        # Expand indices for batch dimension: (1, output_size, n) -> (batch_size, output_size, n)
        indices_expanded = self._mapping_indices.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Convert to long for indexing
        indices_long = indices_expanded.long()
        
        # Gather values: for each (batch, output, pos), get x[batch, indices[output, pos]]
        # x: (batch_size, input_size)
        # We need to gather from input_size dimension using indices
        # Expand x to match output dimension, then gather along input dimension
        
        # Approach: Use batched index_select equivalent via gather
        # x.unsqueeze(1): (batch_size, 1, input_size)
        # expand to: (batch_size, output_size, input_size)
        # then gather along dim=2 using indices: (batch_size, output_size, n)
        x_expanded = x.unsqueeze(1).expand(-1, self.output_size, -1)
        mapped_inputs = torch.gather(x_expanded, dim=2, index=indices_long)
        
        return mapped_inputs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through random mapping and node.
        
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
        # Try CUDA kernel first using the autograd-enabled wrapper
        if self.node.use_cuda and x.is_cuda and _PROBABILISTIC_CUDA_AVAILABLE:
            try:
                # Import the autograd-enabled function from probabilistic_node
                from ..nodes.probabilistic_node import ProbabilisticFunction
                # Ensure temperature is a float (not tensor) for ProbabilisticFunction
                temp_float = float(temperature) if isinstance(temperature, torch.Tensor) else temperature
                output = ProbabilisticFunction.apply(x.contiguous(), raw_weights.contiguous(), temp_float)
                return output
            except Exception as e:
                warnings.warn(
                    f"CUDA kernel failed, falling back to CPU: {e}",
                    RuntimeWarning,
                    stacklevel=2
                )
        
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
        """Get the random mapping matrix for inspection (as indices)."""
        # Return the index mapping directly
        return self._mapping_indices.long()
    
    def extra_repr(self) -> str:
        """String representation for print(model)."""
        flip_str = f", flip_prob={self.flip_probability}" if self.flip_probability > 0 else ""
        grad_str = f", grad_stab={self.grad_stabilization}" if self.grad_stabilization != 'none' else ""
        return f"input_size={self.input_size}, output_size={self.output_size}, " \
               f"n={self.n}, seed={self.seed}, mapping=random{flip_str}{grad_str}"