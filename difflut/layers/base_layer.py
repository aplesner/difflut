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
        
        # Create nodes - input_dim and output_dim should be specified in node_kwargs
        self.nodes = nn.ModuleList()
        for i in range(output_size):
            node_kwargs_i = node_kwargs or {}
            node = node_type(**node_kwargs_i)
            self.nodes.append(node)
        
        # Extract n (number of inputs per node) from the first node
        if len(self.nodes) > 0:
            self.n = self.nodes[0].num_inputs
        else:
            self.n = 0
    
    @abstractmethod
    def get_mapping(self, x: torch.Tensor) -> torch.Tensor:
        """Get mapped inputs for nodes"""
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer
        """
        # Get mapped inputs: (batch_size, output_size, n)
        mapped_inputs = self.get_mapping(x)
        
        # Send all mapped inputs as a 3D tensor directly to nodes
        # Shape: (batch_size, layer_size, input_dim)
        # where layer_size = output_size (number of independent node copies)
        # and input_dim = n (inputs per node)
        output = self.nodes[0](mapped_inputs) if len(self.nodes) > 0 else mapped_inputs
        
        # Output shape: (batch_size, layer_size, output_dim)
        return output
    
    def regularization(self) -> torch.Tensor:
        """Compute regularization for all nodes"""
        reg = torch.tensor(0.0, device=next(self.parameters()).device if list(self.parameters()) else 'cpu')
        for node in self.nodes:
            if hasattr(node, 'regularization'):
                reg = reg + node.regularization()
        return reg / len(self.nodes) if self.nodes else reg