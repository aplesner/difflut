import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional, Type, Tuple, Callable, List
import warnings
from ..nodes.node_config import NodeConfig
from ..utils.warnings import warn_default_value

# Default bit-flip probability for training augmentation
# If flip_probability > 0, randomly flip this fraction of bits during training
DEFAULT_LAYER_FLIP_PROBABILITY: float = 0.0
# Default gradient stabilization mode
# Options: 'none', 'layerwise', 'batchwise'
DEFAULT_LAYER_GRAD_STABILIZATION: str = 'none'
# Default target standard deviation for gradient stabilization
# Used when grad_stabilization is not 'none'
DEFAULT_LAYER_GRAD_TARGET_STD: float = 1.0
# Default flag for whether to subtract mean during gradient stabilization
DEFAULT_LAYER_GRAD_SUBTRACT_MEAN: bool = False
# Default epsilon for numerical stability in gradient stabilization
DEFAULT_LAYER_GRAD_EPSILON: float = 1e-8
# Threshold for warning about excessive input reuse
# Warn if total connections > input_size * LAYER_REUSE_WARNING_THRESHOLD
LAYER_REUSE_WARNING_THRESHOLD: int = 10
# Threshold for warning about underutilized inputs
# Warn if total connections < input_size // LAYER_UNDERUSE_WARNING_DIVISOR
LAYER_UNDERUSE_WARNING_DIVISOR: int = 10
# Maximum recommended node input dimension before warning
LAYER_MAX_NODE_INPUT_DIM: int = 15



class BaseLUTLayer(nn.Module, ABC):
    """
    Base class for LUT layers with proper gradient flow.
    
    Dimension Specification:
    - Input: (batch_size, input_size)
    - Output: (batch_size, output_size * output_dim_per_node)
    
    Each node processes: (batch_size, node_input_dim) → (batch_size, output_dim_per_node)
    The layer uses nn.ModuleList with output_size independent node instances.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        node_type: Type[nn.Module],
        node_kwargs: NodeConfig,
        flip_probability: Optional[float] = None,
        grad_stabilization: Optional[str] = None,
        grad_target_std: Optional[float] = None,
        grad_subtract_mean: Optional[bool] = None,
        grad_epsilon: Optional[float] = None
    ) -> None:
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
            warn_default_value("flip_probability", self.flip_probability, stacklevel=2)
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
            warn_default_value("grad_stabilization", self.grad_stabilization, stacklevel=2)
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
            warn_default_value("grad_target_std", self.grad_target_std, stacklevel=2)
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
            warn_default_value("grad_subtract_mean", self.grad_subtract_mean, stacklevel=2)
        else:
            self.grad_subtract_mean = grad_subtract_mean
        
        # Set grad_epsilon with default
        if grad_epsilon is None:
            self.grad_epsilon = DEFAULT_LAYER_GRAD_EPSILON
            warn_default_value("grad_epsilon", self.grad_epsilon, stacklevel=2)
        else:
            self.grad_epsilon = grad_epsilon
        
        if not isinstance(self.grad_epsilon, (int, float)) or self.grad_epsilon <= 0:
            raise ValueError(
                f"grad_epsilon must be a positive number, got {self.grad_epsilon}. "
                f"Used for numerical stability in variance calculation"
            )
        

        self.input_size = input_size
        self.output_size = output_size
        
        # Create a ModuleList of individual node instances
        # Each node operates independently on (batch_size, input_dim) -> (batch_size, output_dim)
        self.nodes = nn.ModuleList([
            node_type(**node_kwargs.to_dict()) 
            for _ in range(output_size)
        ])
        
        # Extract n (number of inputs per node) and output_dim from first node
        self.n = self.nodes[0].num_inputs
        self.output_dim_per_node = self.nodes[0].num_outputs
        
        # Warn if configuration seems unusual
        self._validate_layer_config()
    
    def _validate_layer_config(self) -> None:
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
    
    def _validate_input_dims(self, x: torch.Tensor) -> None:
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
        """
        Get mapped inputs for all nodes.
        
        Returns:
            3D tensor of shape (batch_size, output_size, n) where each node gets its
            n-dimensional mapped input. This 3D structure is kept for memory efficiency
            even though each node processes independently.
        """
        pass
    
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
        Forward pass through the layer with multiple independent nodes.
        
        Accepts 2D input (batch_size, input_size) and maps it to 
        (batch_size, output_size, n) where each of the output_size nodes processes
        its n-dimensional input independently. Each node in self.nodes ModuleList
        processes (batch_size, n) -> (batch_size, output_dim_per_node).
        
        During training, applies bit-flip augmentation if flip_probability > 0.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
               - From Encoder: (batch_size, encoded_dim)
               - From previous Layer: (batch_size, previous_output_size * previous_output_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_size * output_dim_per_node)
            - For next Layer: (batch_size, output_size * output_dim_per_node)
            - For GroupSum: (batch_size, output_size) if output_dim_per_node=1
        """
        # Validate input dimensions
        self._validate_input_dims(x)
        
        # Apply bit-flip augmentation during training
        if self.training and self.flip_probability > 0.0:
            x = self._apply_bit_flip(x)
        
        # Get mapped inputs: (batch_size, output_size, n)
        # where n = node.num_inputs
        mapped_inputs = self.get_mapping(x)
        
        batch_size = mapped_inputs.shape[0]
        
        # Process each node independently
        # Each node: (batch_size, n) -> (batch_size, output_dim_per_node)
        # Stack outputs: (batch_size, output_size, output_dim_per_node)
        node_outputs = []
        for node_idx, node in enumerate(self.nodes):
            # Extract mapped input for this node: (batch_size, n)
            node_input = mapped_inputs[:, node_idx, :]
            # Process through node: (batch_size, n) -> (batch_size, output_dim_per_node)
            node_output = node(node_input)
            node_outputs.append(node_output)
        
        # Stack along node dimension: (batch_size, output_size, output_dim_per_node)
        output = torch.stack(node_outputs, dim=1)
        
        # Reshape to 2D for next layer: (batch_size, output_size * output_dim_per_node)
        # In most cases output_dim_per_node=1, so this flattens to (batch_size, output_size)
        output = output.view(batch_size, -1)
        
        # Register gradient stabilization hook if enabled
        if self.grad_stabilization != 'none' and self.training and output.requires_grad:
            output.register_hook(self._apply_gradient_stabilization)
        
        return output
    
    def regularization(self) -> torch.Tensor:
        """Compute total regularization across all nodes"""
        total_reg = torch.tensor(0.0, device=next(self.parameters()).device if list(self.parameters()) else 'cpu')
        
        for node in self.nodes:
            if hasattr(node, 'regularization'):
                total_reg = total_reg + node.regularization()
        
        return total_reg