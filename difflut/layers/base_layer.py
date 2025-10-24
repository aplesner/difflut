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
                 node_kwargs=None):
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
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Create a single node - it will handle parallelization via CUDA kernels
        # treating output_size as an additional batch dimension
        node_kwargs = node_kwargs or {}
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer.
        
        Accepts 2D input (batch_size, input_size) and maps it to 
        (batch_size, output_size, node.num_inputs) before passing to the single node.
        The node handles parallelization across output_size using CUDA kernels.
        
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
        
        return output
    
    def regularization(self) -> torch.Tensor:
        """Compute regularization for the single node"""
        if hasattr(self.node, 'regularization'):
            return self.node.regularization()
        else:
            return torch.tensor(0.0, device=next(self.node.parameters()).device if list(self.node.parameters()) else 'cpu')