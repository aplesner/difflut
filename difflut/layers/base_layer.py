import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Union, Dict, Any
import warnings
from ..constants import (
    LAYER_REUSE_WARNING_THRESHOLD,
    LAYER_UNDERUSE_WARNING_DIVISOR,
    LAYER_MAX_NODE_INPUT_DIM,
    DEFAULT_LAYER_FLIP_PROBABILITY,
    DEFAULT_LAYER_GRAD_STABILIZATION,
    DEFAULT_LAYER_GRAD_TARGET_STD,
    DEFAULT_LAYER_GRAD_SUBTRACT_MEAN,
    DEFAULT_LAYER_GRAD_EPSILON,
    DEFAULT_LAYER_MAX_NODES_PER_BATCH
)
from ..nodes.node_config import NodeConfig, NodeKwargs, normalize_node_kwargs


class BaseLUTLayer(nn.Module, ABC):
    """
    Base class for LUT layers with proper gradient flow.
    
    Dimension Specification:
    - Input: (batch_size, input_size)
    - Output: (batch_size, num_nodes * num_output_per_node)
    - Internal: (batch_size, input_size) → (batch_size, num_nodes, node_input_dim)
    
    The layer maps 2D input to 3D node inputs, processes through nodes,
    and reshapes back to 2D output for the next layer.
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 node_type,
                 node_kwargs: NodeKwargs = None,
                 flip_probability: float = None,
                 grad_stabilization: str = None,
                 grad_target_std: float = None,
                 grad_subtract_mean: bool = None,
                 grad_epsilon: float = None,
                 max_nodes_per_batch: int = None):
        super().__init__()
        
        # Validate parameters
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError(
                f"input_size must be a positive integer, got {input_size}. "
                f"This typically comes from an encoder output or previous layer output."
            )
        
        if not isinstance(output_size, int) or output_size <= 0:
            raise ValueError(
                f"output_size must be a positive integer, got {output_size}. "
                f"This is the number of nodes in the layer."
            )
        
        # Set flip_probability with default
        if flip_probability is None:
            self.flip_probability = DEFAULT_LAYER_FLIP_PROBABILITY
        else:
            self.flip_probability = flip_probability
        
        # Validate flip_probability
        if not isinstance(self.flip_probability, (int, float)) or not (0.0 <= self.flip_probability <= 1.0):
            raise ValueError(
                f"flip_probability must be a float in [0, 1], got {self.flip_probability}. "
                f"Example: flip_probability=0.1 for 10% bit flipping during training."
            )
        
        # Set grad_stabilization with default
        if grad_stabilization is None:
            self.grad_stabilization = DEFAULT_LAYER_GRAD_STABILIZATION
        else:
            self.grad_stabilization = grad_stabilization
        
        # Validate gradient stabilization parameters
        valid_grad_modes = ['none', 'layerwise', 'batchwise']
        if self.grad_stabilization not in valid_grad_modes:
            raise ValueError(
                f"grad_stabilization must be one of {valid_grad_modes}, got '{self.grad_stabilization}'. "
                f"'layerwise': normalize per layer, 'batchwise': normalize per batch sample, 'none': disabled"
            )
        
        # Set grad_target_std with default
        if grad_target_std is None:
            self.grad_target_std = DEFAULT_LAYER_GRAD_TARGET_STD
        else:
            self.grad_target_std = grad_target_std
        
        if not isinstance(self.grad_target_std, (int, float)) or self.grad_target_std <= 0:
            raise ValueError(
                f"grad_target_std must be a positive number, got {self.grad_target_std}. "
                f"Example: grad_target_std=1.0 for unit variance"
            )
        
        # Set grad_subtract_mean with default
        if grad_subtract_mean is None:
            self.grad_subtract_mean = DEFAULT_LAYER_GRAD_SUBTRACT_MEAN
        else:
            self.grad_subtract_mean = grad_subtract_mean
        
        # Set grad_epsilon with default
        if grad_epsilon is None:
            self.grad_epsilon = DEFAULT_LAYER_GRAD_EPSILON
        else:
            self.grad_epsilon = grad_epsilon
        
        if not isinstance(self.grad_epsilon, (int, float)) or self.grad_epsilon <= 0:
            raise ValueError(
                f"grad_epsilon must be a positive number, got {self.grad_epsilon}. "
                f"Used for numerical stability in variance calculation"
            )
        
        # Set max_nodes_per_batch with default
        if max_nodes_per_batch is None:
            self.max_nodes_per_batch = DEFAULT_LAYER_MAX_NODES_PER_BATCH
        else:
            self.max_nodes_per_batch = max_nodes_per_batch
        
        if not isinstance(self.max_nodes_per_batch, int) or (self.max_nodes_per_batch <= 0 and self.max_nodes_per_batch != -1):
            raise ValueError(
                f"max_nodes_per_batch must be a positive integer or -1 (disable batching), got {self.max_nodes_per_batch}. "
                f"Recommended: 256 (low memory), 512 (balanced), 1024 (high memory), -1 (no batching)"
            )
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Create nodes with layer_size parameter - each position gets its own parameters
        # No weight sharing across layer dimension
        # Normalize node_kwargs to NodeConfig for type safety and maintainability
        node_config = normalize_node_kwargs(node_kwargs)
        node_config_with_layer = node_config.with_layer_size(output_size)
        self.node = node_type(**node_config_with_layer.to_dict())
        
        # Extract n (number of inputs per node)
        self.n = self.node.num_inputs
        
        # Memory optimization: preallocate buffer for bit-flip mask (reused across forward passes)
        self.register_buffer('_flip_mask_buffer', None)
        
        # Warn if configuration seems unusual
        self._validate_layer_config()
    
    def _validate_layer_config(self):
        """
        Validate that layer configuration makes sense.
        Generate warnings for unusual but valid configurations.
        """
        total_connections = self.output_size * self.n
        
        # Warning 1: Very large mapping
        if total_connections > self.input_size * LAYER_REUSE_WARNING_THRESHOLD:
            warnings.warn(
                f"BaseLUTLayer: Creating {total_connections} node input connections from only "
                f"{self.input_size} input features. Each input feature will be reused "
                f"{total_connections // self.input_size}x on average. This may lead to overfitting. "
                f"Consider using more input features or fewer nodes (output_size={self.output_size}, n={self.n}).",
                UserWarning,
                stacklevel=2
            )
        
        # Warning 2: Very small mapping
        if self.output_size * self.n < self.input_size // LAYER_UNDERUSE_WARNING_DIVISOR:
            warnings.warn(
                f"BaseLUTLayer: Creating only {total_connections} node inputs from "
                f"{self.input_size} input features. Most input features will be unused. "
                f"Consider using more nodes (output_size={self.output_size}) or larger node input dimension (n={self.n}).",
                UserWarning,
                stacklevel=2
            )
        
        # Warning 3: Large node input dimension
        if self.n > LAYER_MAX_NODE_INPUT_DIM:
            warnings.warn(
                f"BaseLUTLayer: Node input dimension (n={self.n}) is quite large. "
                f"LUT nodes with >15 inputs may have exponentially large memory requirements (2^{self.n} entries). "
                f"Consider reducing node input dimension or splitting across more layers.",
                UserWarning,
                stacklevel=2
            )
    
    def _validate_input_dims(self, x: torch.Tensor):
        """
        Validate that input has expected dimensions.
        
        Args:
            x: Input tensor
            
        Raises:
            ValueError: If input dimensions are invalid
        """
        if x.dim() != 2:
            raise ValueError(
                f"BaseLUTLayer expects 2D input (batch_size, input_size), "
                f"but got shape {x.shape} with {x.dim()} dimensions. "
                f"Input should come from an Encoder (batch_size, encoded_dim) "
                f"or previous Layer (batch_size, num_nodes * num_output_per_node)."
            )
        
        batch_size, feat_size = x.shape
        
        if feat_size != self.input_size:
            raise ValueError(
                f"BaseLUTLayer expected input with {self.input_size} features, "
                f"but got {feat_size} features. Shape: {x.shape}. "
                f"Ensure that the input source (Encoder or previous Layer) outputs exactly "
                f"{self.input_size} features."
            )
        
        if batch_size == 0:
            raise ValueError(
                f"BaseLUTLayer requires non-empty batch, got batch_size={batch_size}"
            )
    
    @abstractmethod
    def get_mapping(self, x: torch.Tensor) -> torch.Tensor:
        """Get mapped inputs for nodes"""
        pass
    
    def _forward_with_node_batching(self, mapped_inputs: torch.Tensor) -> torch.Tensor:
        """
        Process nodes in batches to reduce memory usage.
        
        Memory optimization for large layers: instead of processing all nodes at once,
        process them in chunks of max_nodes_per_batch. This prevents OOM errors on
        GPUs with limited memory when output_size is very large (e.g., 10,000+ nodes).
        
        The key insight: nodes have per-layer-node parameters stored as tensors with
        layer_size as the first dimension (e.g., raw_weights shape: (layer_size, 2^n, output_dim)).
        We slice both the input and the parameters for each chunk, then call the node's
        forward method with torch.no_grad() temporarily disabled so gradients flow correctly.
        
        Args:
            mapped_inputs: (batch_size, output_size, n) tensor
        
        Returns:
            output: (batch_size, output_size, output_dim) tensor
        """
        from ..nodes.probabilistic_node import ProbabilisticNode
        
        batch_size, output_size, n = mapped_inputs.shape
        
        # Preallocate output tensor
        output = torch.empty(
            (batch_size, output_size, self.node.output_dim),
            device=mapped_inputs.device,
            dtype=mapped_inputs.dtype
        )
        
        # Check if node is ProbabilisticNode (has raw_weights parameter)
        is_probabilistic = isinstance(self.node, ProbabilisticNode)
        
        # Process nodes in chunks
        for start_idx in range(0, output_size, self.max_nodes_per_batch):
            end_idx = min(start_idx + self.max_nodes_per_batch, output_size)
            
            # Extract chunk: (batch_size, chunk_size, n)
            mapped_chunk = mapped_inputs[:, start_idx:end_idx, :]
            
            # Process chunk through node with parameter slicing
            if is_probabilistic:
                # For ProbabilisticNode, we need to slice raw_weights and call forward_train directly
                raw_weights_chunk = self.node.raw_weights[start_idx:end_idx]
                
                # Call forward_train with sliced weights
                if self.training:
                    # Use the probabilistic forward with sliced parameters
                    if self.node.use_cuda and mapped_chunk.is_cuda:
                        from ..nodes.probabilistic_node import probabilistic_forward
                        output_chunk = probabilistic_forward(
                            mapped_chunk, 
                            raw_weights_chunk, 
                            self.node.temperature
                        )
                    else:
                        # CPU fallback - need to implement manual computation
                        output_chunk = self._probabilistic_forward_cpu(
                            mapped_chunk, 
                            raw_weights_chunk, 
                            self.node.temperature, 
                            self.node.weights[start_idx:end_idx],
                            self.node.binary_combinations
                        )
                else:
                    # Evaluation mode - use forward_eval
                    output_chunk = self.node.forward_eval(mapped_chunk)
            else:
                # For other node types, just call forward
                # This may not work correctly if they also have per-layer parameters
                output_chunk = self.node(mapped_chunk)
            
            # Store result
            output[:, start_idx:end_idx, :] = output_chunk
        
        return output
    
    def _probabilistic_forward_cpu(
        self, 
        x: torch.Tensor, 
        raw_weights: torch.Tensor,
        temperature: torch.Tensor,
        weights: torch.Tensor,
        binary_combinations: torch.Tensor
    ) -> torch.Tensor:
        """
        CPU fallback for probabilistic forward with sliced parameters.
        
        Args:
            x: (batch_size, chunk_size, input_dim) tensor
            raw_weights: (chunk_size, 2^input_dim, output_dim) tensor
            temperature: scalar tensor
            weights: (chunk_size, 2^input_dim, output_dim) tensor (sigmoid applied)
            binary_combinations: (2^input_dim, input_dim) tensor
        
        Returns:
            output: (batch_size, chunk_size, output_dim) tensor
        """
        batch_size, chunk_size, input_dim = x.shape
        output_dim = weights.shape[2]
        
        # Ensure binary_combinations is on the same device
        binary_combinations = binary_combinations.to(device=x.device, dtype=x.dtype)
        
        # Preallocate output
        output = torch.empty(
            (batch_size, chunk_size, output_dim),
            device=x.device,
            dtype=x.dtype
        )
        
        # Process each layer independently
        for layer_idx in range(chunk_size):
            x_layer = x[:, layer_idx, :]  # (batch_size, input_dim)
            
            # Vectorized probability computation
            x_expanded = x_layer.unsqueeze(1)  # (batch_size, 1, input_dim)
            a_expanded = binary_combinations.unsqueeze(0)  # (1, 2^input_dim, input_dim)
            prob_terms = x_expanded * a_expanded + (1 - x_expanded) * (1 - a_expanded)
            probs = torch.prod(prob_terms, dim=-1)  # (batch_size, 2^input_dim)
            
            # Apply weights
            output[:, layer_idx, :] = torch.matmul(probs, weights[layer_idx])
        
        return output
    
    def _apply_bit_flip(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply bit-flip augmentation during training (memory-optimized, gradient-detached).
        Randomly flips flip_probability fraction of bits (x -> 1-x).
        
        Gradient Behavior:
        - Forward: Model sees flipped bits (adds noise/corruption)
        - Backward: Gradients flow as if no flip occurred (∂L/∂x based on original x)
        
        This treats bit-flipping as **pure noise injection** for robustness training.
        The model learns to be robust to corruption, not to predict and undo it.
        Gradients are not contaminated by random noise from the flip operation.
        
        Memory optimization: Uses preallocated buffer for mask generation and
        sparse indexing for low flip probabilities to minimize memory allocations.
        
        Args:
            x: Input tensor of shape (batch_size, input_size) with values in [0, 1]
        
        Returns:
            Augmented tensor with same shape (flipped bits detached from gradient graph)
        """
        if self.flip_probability <= 0.0 or not self.training:
            return x
        
        # For very low flip probabilities, use sparse indexing (more efficient)
        if self.flip_probability < 0.05:
            return self._apply_bit_flip_sparse(x)
        
        # Standard approach with buffer reuse for moderate-to-high probabilities
        batch_size, input_size = x.shape
        
        # Preallocate or reuse buffer (amortize allocation cost)
        if (self._flip_mask_buffer is None or 
            self._flip_mask_buffer.shape[0] < batch_size or
            self._flip_mask_buffer.shape[1] < input_size):
            # Allocate buffer large enough for future batches
            buffer_batch = max(batch_size, 256)  # Support up to 256 batch size
            buffer_features = input_size
            self._flip_mask_buffer = torch.empty(
                (buffer_batch, buffer_features),
                dtype=torch.bool,
                device=x.device
            )
        
        # Get view of buffer matching current batch (no allocation)
        mask = self._flip_mask_buffer[:batch_size, :input_size]
        
        # Generate random mask in-place (reuses buffer memory)
        # Note: bernoulli with tensor input requires float for output
        mask_float = mask.float()
        torch.bernoulli(
            torch.full((batch_size, input_size), self.flip_probability, 
                      device=x.device, dtype=torch.float32),
            out=mask_float
        )
        mask = mask_float.bool()
        
        # OPTIMIZED: Compute flip delta without cloning
        # Only create tensor for the difference (sparse operation)
        # Delta: (1-x) - x = 1 - 2x for flipped positions
        flip_delta = torch.zeros_like(x)
        flip_delta[mask] = (1.0 - 2.0 * x[mask])
        
        # CRITICAL: Detach noise from gradient graph
        # Forward: Model sees flipped bits (x + delta = x + (1-2x) = 1-x for masked positions)
        # Backward: Gradients flow as if no flip occurred (∂L/∂x based on original x)
        # This treats bit-flipping as pure noise injection for robustness training,
        # not as a learnable transformation that the model should compensate for.
        # The model learns: "be robust to corruption" not "predict and undo corruption"
        x_out = x + flip_delta.detach()
        
        return x_out
    
    def _apply_bit_flip_sparse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sparse bit flipping for low probabilities (<5%).
        Uses random sampling to flip only the required number of elements.
        More memory efficient when flip_probability is small.
        
        Gradient behavior: Same as standard bit-flip (gradient-detached noise).
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Augmented tensor with same shape (flipped bits detached from gradient graph)
        """
        num_elements = x.numel()
        num_flips = int(num_elements * self.flip_probability)
        
        if num_flips == 0:
            return x
        
        # OPTIMIZED: Compute flip delta without cloning
        # Only create tensor for the difference (sparse operation)
        flip_delta = torch.zeros_like(x)
        flat_delta = flip_delta.view(-1)
        flat_x = x.view(-1)
        
        # Random sample without replacement (select indices to flip)
        flip_indices = torch.randperm(num_elements, device=x.device)[:num_flips]
        
        # Delta: (1-x) - x = 1 - 2x for flipped positions
        flat_delta[flip_indices] = 1.0 - 2.0 * flat_x[flip_indices]
        
        # CRITICAL: Detach noise from gradient graph (same as standard bit-flip)
        # Forward: Model sees flipped bits (x + delta)
        # Backward: Gradients ignore the flip (robustness training)
        x_out = x + flip_delta.detach()
        
        return x_out
    
    def _apply_gradient_stabilization(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Apply gradient stabilization (rescaling) to normalize gradient variance.
        Memory-optimized version using in-place operations where possible.
        
        Implements layer-wise or batch-wise gradient rescaling as described in:
        Definition [Layer-wise Gradient Rescaling]:
        
        For layer gradients ∇c^l, compute variance v_l and optionally mean μ_l,
        then rescale: ∇c_i^l ← (∇c_i^l - μ_l) / √(v_l + ε) · √v_target
        
        Args:
            grad: Gradient tensor of shape (batch_size, output_size)
                 For layer output before it's reshaped
        
        Returns:
            Rescaled gradient with same shape
        """
        if self.grad_stabilization == 'none' or not self.training:
            return grad
        
        if grad is None:
            return grad
        
        # Clone to avoid modifying in-place during autograd
        grad_work = grad.clone()
        
        if self.grad_stabilization == 'layerwise':
            # Layer-wise: normalize across all elements in the layer
            # Shape: (batch_size, output_size) → treat as one layer
            
            # Compute and subtract mean (optional, in-place)
            if self.grad_subtract_mean:
                mu = grad_work.mean()
                grad_work.sub_(mu)  # In-place subtraction
            
            # Compute variance
            variance = grad_work.pow(2).mean()
            
            # Compute scale factor
            scale = torch.sqrt(torch.tensor(self.grad_target_std, device=grad.device) / 
                             (variance + self.grad_epsilon))
            
            # Rescale in-place
            grad_work.mul_(scale)
            
            return grad_work
        
        elif self.grad_stabilization == 'batchwise':
            # Batch-wise: normalize per sample across the layer dimension
            # Shape: (batch_size, output_size) → normalize each batch element independently
            
            # Compute and subtract mean per batch sample (optional, in-place)
            if self.grad_subtract_mean:
                mu = grad_work.mean(dim=1, keepdim=True)  # (batch_size, 1)
                grad_work.sub_(mu)  # In-place subtraction
            
            # Compute variance per batch sample
            variance = grad_work.pow(2).mean(dim=1, keepdim=True)  # (batch_size, 1)
            
            # Compute scale factor
            scale = torch.sqrt(torch.tensor(self.grad_target_std, device=grad.device) / 
                             (variance + self.grad_epsilon))
            
            # Rescale in-place
            grad_work.mul_(scale)
            
            return grad_work
        
        return grad
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer.
        
        Accepts 2D input (batch_size, input_size) and maps it to 
        (batch_size, output_size, node.num_inputs) before passing to the single node.
        The node handles parallelization across output_size using CUDA kernels.
        
        During training, applies bit-flip augmentation if flip_probability > 0.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
               - From Encoder: (batch_size, encoded_dim)
               - From previous Layer: (batch_size, previous_output_size * previous_output_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_size * output_dim)
            - For next Layer: (batch_size, output_size * output_dim)
            - For GroupSum: (batch_size, output_size) if output_dim=1
        """
        # Validate input dimensions
        self._validate_input_dims(x)
        
        # Apply bit-flip augmentation during training
        if self.training and self.flip_probability > 0.0:
            x = self._apply_bit_flip(x)
        
        # Get mapped inputs: (batch_size, output_size, n)
        # where n = node.num_inputs
        mapped_inputs = self.get_mapping(x)
        
        # MEMORY OPTIMIZATION: Process nodes in batches if output_size is large
        # This prevents OOM errors by chunking the layer dimension
        batch_size = mapped_inputs.shape[0]
        
        if self.max_nodes_per_batch > 0 and self.output_size > self.max_nodes_per_batch and self.training:
            # Layer-wise batching: process nodes in chunks
            output = self._forward_with_node_batching(mapped_inputs)
        else:
            # Standard path: process all nodes at once
            # Shape: (batch_size, output_size, n) -> (batch_size, output_size, output_dim)
            output = self.node(mapped_inputs)
        
        # Output shape: (batch_size, output_size, output_dim)
        # Reshape to 2D for next layer: (batch_size, output_size * output_dim)
        # In most cases output_dim=1, so this just squeezes out the last dimension
        output = output.view(batch_size, -1)
        
        # Register gradient stabilization hook if enabled
        if self.grad_stabilization != 'none' and self.training and output.requires_grad:
            output.register_hook(self._apply_gradient_stabilization)
        
        return output
    
    def regularization(self) -> torch.Tensor:
        """Compute regularization for the single node"""
        if hasattr(self.node, 'regularization'):
            return self.node.regularization()
        else:
            return torch.tensor(0.0, device=next(self.node.parameters()).device if list(self.node.parameters()) else 'cpu')