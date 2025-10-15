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
                 n: int = 6,
                 node_kwargs=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n = n
        
        # Create nodes with new input_dim and output_dim interface
        self.nodes = nn.ModuleList()
        for i in range(output_size):
            node_kwargs_i = node_kwargs or {}
            # Use new interface: input_dim and output_dim as lists
            node = node_type(input_dim=[n], output_dim=[1], **node_kwargs_i)
            self.nodes.append(node)
    
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
        
        # Process through nodes
        outputs = []
        for i, node in enumerate(self.nodes):
            # Get inputs for this node
            node_input = mapped_inputs[:, i, :]  # Shape: (batch_size, n)
            # Add a dummy dimension if node expects it
            if len(node_input.shape) == 2:
                node_input = node_input.unsqueeze(1)  # Shape: (batch_size, 1, n)
            output = node(node_input)
            # Remove extra dimensions if present
            if len(output.shape) > 2:
                output = output.squeeze(1)
            outputs.append(output)
        
        # Stack outputs
        return torch.stack(outputs, dim=1).squeeze(-1) if outputs[0].dim() > 1 else torch.stack(outputs, dim=1)
    
    def regularization(self) -> torch.Tensor:
        """Compute regularization for all nodes"""
        reg = torch.tensor(0.0, device=next(self.parameters()).device if list(self.parameters()) else 'cpu')
        for node in self.nodes:
            if hasattr(node, 'regularization'):
                reg = reg + node.regularization()
        return reg / len(self.nodes) if self.nodes else reg