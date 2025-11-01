import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import warnings


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
                 node_kwargs=None,
                 flip_probability: float = 0.0,
                 grad_stabilization: str = 'none',
                 grad_target_std: float = 1.0,
                 grad_subtract_mean: bool = False,
                 grad_epsilon: float = 1e-8):
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
        
        # Validate flip_probability
        if not isinstance(flip_probability, (int, float)) or not (0.0 <= flip_probability <= 1.0):
            raise ValueError(
                f"flip_probability must be a float in [0, 1], got {flip_probability}. "
                f"Example: flip_probability=0.1 for 10% bit flipping during training."
            )
        
        # Validate gradient stabilization parameters
        valid_grad_modes = ['none', 'layerwise', 'batchwise']
        if grad_stabilization not in valid_grad_modes:
            raise ValueError(
                f"grad_stabilization must be one of {valid_grad_modes}, got '{grad_stabilization}'. "
                f"'layerwise': normalize per layer, 'batchwise': normalize per batch sample, 'none': disabled"
            )
        
        if not isinstance(grad_target_std, (int, float)) or grad_target_std <= 0:
            raise ValueError(
                f"grad_target_std must be a positive number, got {grad_target_std}. "
                f"Example: grad_target_std=1.0 for unit variance"
            )
        
        if not isinstance(grad_epsilon, (int, float)) or grad_epsilon <= 0:
            raise ValueError(
                f"grad_epsilon must be a positive number, got {grad_epsilon}. "
                f"Used for numerical stability in variance calculation"
            )
        
        self.input_size = input_size
        self.output_size = output_size
        self.flip_probability = flip_probability
        self.grad_stabilization = grad_stabilization
        self.grad_target_std = grad_target_std
        self.grad_subtract_mean = grad_subtract_mean
        self.grad_epsilon = grad_epsilon
        
        # Create nodes with layer_size parameter - each position gets its own parameters
        # No weight sharing across layer dimension
        node_kwargs = node_kwargs or {}
        node_kwargs['layer_size'] = output_size  # Pass layer_size to node
        self.node = node_type(**node_kwargs)
        
        # Extract n (number of inputs per node)
        self.n = self.node.num_inputs
        
        # Warn if configuration seems unusual
        self._validate_layer_config()
    
    def _validate_layer_config(self):
        """
        Validate that layer configuration makes sense.
        Generate warnings for unusual but valid configurations.
        """
        total_connections = self.output_size * self.n
        
        # Warning 1: Very large mapping
        if total_connections > self.input_size * 100:
            warnings.warn(
                f"BaseLUTLayer: Creating {total_connections} node input connections from only "
                f"{self.input_size} input features. Each input feature will be reused "
                f"{total_connections // self.input_size}x on average. This may lead to overfitting. "
                f"Consider using more input features or fewer nodes (output_size={self.output_size}, n={self.n}).",
                UserWarning,
                stacklevel=2
            )
        
        # Warning 2: Very small mapping
        if self.output_size * self.n < self.input_size // 10:
            warnings.warn(
                f"BaseLUTLayer: Creating only {total_connections} node inputs from "
                f"{self.input_size} input features. Most input features will be unused. "
                f"Consider using more nodes (output_size={self.output_size}) or larger node input dimension (n={self.n}).",
                UserWarning,
                stacklevel=2
            )
        
        # Warning 3: Large node input dimension
        if self.n > 15:
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
    
    def _apply_bit_flip(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply bit-flip augmentation during training.
        Randomly flips flip_probability fraction of bits (x -> 1-x).
        
        Args:
            x: Input tensor of shape (batch_size, input_size) with values in [0, 1]
        
        Returns:
            Augmented tensor with same shape
        """
        if self.flip_probability <= 0.0 or not self.training:
            return x
        
        # Create random mask: True where we should flip
        flip_mask = torch.rand_like(x) < self.flip_probability
        
        # Flip selected bits: x -> 1 - x
        x_flipped = torch.where(flip_mask, 1.0 - x, x)
        
        return x_flipped
    
    def _apply_gradient_stabilization(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Apply gradient stabilization (rescaling) to normalize gradient variance.
        
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
        
        if self.grad_stabilization == 'layerwise':
            # Layer-wise: normalize across all elements in the layer
            # Shape: (batch_size, output_size) → treat as one layer
            
            # Compute mean (optional)
            if self.grad_subtract_mean:
                mu = grad.mean()
                grad_centered = grad - mu
            else:
                grad_centered = grad
            
            # Compute variance
            variance = (grad_centered ** 2).mean()
            
            # Rescale: normalize to unit variance, then scale to target
            grad_rescaled = grad_centered / torch.sqrt(variance + self.grad_epsilon) * torch.sqrt(torch.tensor(self.grad_target_std))
            
            return grad_rescaled
        
        elif self.grad_stabilization == 'batchwise':
            # Batch-wise: normalize per sample across the layer dimension
            # Shape: (batch_size, output_size) → normalize each batch element independently
            
            # Compute mean per batch sample (optional)
            if self.grad_subtract_mean:
                mu = grad.mean(dim=1, keepdim=True)  # (batch_size, 1)
                grad_centered = grad - mu
            else:
                grad_centered = grad
            
            # Compute variance per batch sample
            variance = (grad_centered ** 2).mean(dim=1, keepdim=True)  # (batch_size, 1)
            
            # Rescale each batch sample independently
            grad_rescaled = grad_centered / torch.sqrt(variance + self.grad_epsilon) * torch.sqrt(torch.tensor(self.grad_target_std))
            
            return grad_rescaled
        
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
        
        # Pass the 3D tensor to the single node
        # Shape: (batch_size, output_size, n)
        # The node treats output_size as an additional batch dimension for parallelization
        output = self.node(mapped_inputs)
        
        # Output shape: (batch_size, output_size, output_dim)
        # Reshape to 2D for next layer: (batch_size, output_size * output_dim)
        # In most cases output_dim=1, so this just squeezes out the last dimension
        batch_size = output.shape[0]
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