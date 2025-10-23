import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseLUTLayer(nn.Module, ABC):
    """
    Base class for LUT layers with proper gradient flow
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 node_type,
                 node_kwargs=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Create a single node - it will handle parallelization via CUDA kernels
        # treating output_size as an additional batch dimension
        node_kwargs = node_kwargs or {}
        self.node = node_type(**node_kwargs)
        
        # Extract n (number of inputs per node)
        self.n = self.node.num_inputs
    
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
        """
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