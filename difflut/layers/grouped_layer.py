import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Dict, Any, Optional
from .base_layer import BaseLUTLayer
from ..registry import register_layer


class GroupedMappingModule(nn.Module):
    """
    Grouped mapping module that reduces parameters by dividing the input space into groups.
    Each group of nodes only learns to connect to a subset of inputs.
    
    This achieves parameter reduction while maintaining learnability:
    - Learnable layer: output_size * n * input_size parameters
    - Grouped layer: output_size * n * group_size parameters (where group_size < input_size)
    """
    
    def __init__(self, 
                 input_size: int, 
                 output_size: int,
                 n_inputs_per_node: int,
                 num_groups: int = 4,
                 tau: float = 0.001):
        """
        Args:
            input_size: Total number of input features
            output_size: Number of nodes (each node needs n connections)
            n_inputs_per_node: Number of inputs each node needs (n)
            num_groups: Number of groups to divide the input space into
            tau: Temperature for softmax during training
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_inputs_per_node = n_inputs_per_node
        self.num_groups = num_groups
        self.tau = tau
        
        # Calculate group size and nodes per group
        self.group_size = (input_size + num_groups - 1) // num_groups  # Ceiling division
        self.nodes_per_group = (output_size + num_groups - 1) // num_groups
        
        # Create weight matrices for each group
        # Each group has nodes_per_group nodes, each selecting from group_size inputs
        self.group_weights = nn.ParameterList([
            nn.Parameter(torch.randn(
                min(self.nodes_per_group, output_size - i * self.nodes_per_group),
                n_inputs_per_node,
                min(self.group_size, input_size - i * self.group_size)
            ))
            for i in range(num_groups)
        ])
        
        # Initialize weights
        for weight in self.group_weights:
            nn.init.xavier_uniform_(weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with grouped connections.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        Returns:
            Output tensor of shape (batch_size, output_size, n_inputs_per_node)
        """
        batch_size = x.shape[0]
        outputs = []
        
        node_offset = 0
        input_offset = 0
        
        for group_idx in range(self.num_groups):
            # Determine the actual sizes for this group
            group_nodes = min(self.nodes_per_group, self.output_size - node_offset)
            group_inputs = min(self.group_size, self.input_size - input_offset)
            
            if group_nodes == 0:
                break
            
            # Get the input slice for this group
            x_group = x[:, input_offset:input_offset + group_inputs]  # (batch_size, group_inputs)
            
            # Get weights for this group
            W = self.group_weights[group_idx]  # (group_nodes, n, group_inputs)
            
            if self.training:
                # Soft selection during training
                weights = F.softmax(W / self.tau, dim=-1)  # (group_nodes, n, group_inputs)
                
                # Compute output: batch matmul
                # x_group: (batch_size, group_inputs)
                # weights: (group_nodes, n, group_inputs)
                # We want: (batch_size, group_nodes, n)
                group_output = torch.einsum('bi,oni->bon', x_group, weights)
            else:
                # Hard selection during evaluation
                hard_indices = torch.argmax(W, dim=-1)  # (group_nodes, n)
                
                # Gather from input
                # Expand x_group for gathering: (batch_size, 1, 1, group_inputs)
                x_expanded = x_group.unsqueeze(1).unsqueeze(1)
                # Expand indices: (1, group_nodes, n, 1)
                indices_expanded = hard_indices.unsqueeze(0).unsqueeze(-1).expand(
                    batch_size, -1, -1, 1
                )
                
                # Gather: (batch_size, group_nodes, n)
                group_output = torch.gather(
                    x_expanded.expand(batch_size, group_nodes, self.n_inputs_per_node, group_inputs),
                    3,
                    indices_expanded
                ).squeeze(-1)
            
            outputs.append(group_output)
            
            node_offset += group_nodes
            input_offset += group_inputs
        
        # Concatenate all group outputs along the node dimension
        full_output = torch.cat(outputs, dim=1)  # (batch_size, output_size, n)
        
        return full_output
    
    def count_parameters(self):
        """Count the number of parameters in this module."""
        return sum(p.numel() for p in self.parameters())


@register_layer("grouped")
class GroupedLayer(BaseLUTLayer):
    """
    Grouped learnable LUT layer with reduced parameters.
    
    This layer divides the input space into groups and assigns nodes to groups.
    Each node only learns to connect to inputs within its assigned group, significantly
    reducing the number of parameters compared to a fully learnable layer.
    
    Parameter comparison:
    - Learnable: output_size * n * input_size
    - Grouped: output_size * n * (input_size / num_groups)
    - Reduction: (num_groups - 1) / num_groups (e.g., 75% with 4 groups)
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int, 
                 node_type: Type[nn.Module],
                 n: int = 6,
                 node_kwargs: Optional[Dict[str, Any]] = None,
                 num_groups: int = 4,
                 tau: float = 0.001):
        """
        Args:
            input_size: Size of input vector
            output_size: Number of LUT nodes
            node_type: LUT node class
            n: Number of inputs per LUT
            node_kwargs: Additional arguments for nodes
            num_groups: Number of groups to divide inputs and nodes into
            tau: Temperature for softmax in grouped mapping
        """
        # Initialize parent with nodes
        super().__init__(input_size, output_size, node_type, n, node_kwargs)
        
        self.num_groups = num_groups
        self.tau = tau
        
        # Create grouped mapping module
        self.mapping = GroupedMappingModule(
            input_size=input_size,
            output_size=output_size,
            n_inputs_per_node=n,
            num_groups=num_groups,
            tau=tau
        )
    
    def get_mapping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply grouped mapping.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        Returns:
            Mapped inputs of shape (batch_size, output_size, n)
        """
        return self.mapping(x)
    
    def get_mapping_matrix(self) -> torch.Tensor:
        """Get current hard mapping (for inspection)."""
        with torch.no_grad():
            mapping_matrix = torch.zeros(self.output_size, self.n, dtype=torch.long)
            
            node_offset = 0
            input_offset = 0
            
            for group_idx in range(self.num_groups):
                group_nodes = min(self.mapping.nodes_per_group, 
                                self.output_size - node_offset)
                group_inputs = min(self.mapping.group_size, 
                                 self.input_size - input_offset)
                
                if group_nodes == 0:
                    break
                
                W = self.mapping.group_weights[group_idx]
                hard_indices = torch.argmax(W, dim=-1)  # (group_nodes, n)
                
                # Add input offset to get global indices
                mapping_matrix[node_offset:node_offset + group_nodes] = \
                    hard_indices + input_offset
                
                node_offset += group_nodes
                input_offset += group_inputs
            
            return mapping_matrix
    
    def count_parameters(self):
        """Count total parameters including nodes and mapping."""
        total = sum(p.numel() for p in self.parameters())
        mapping_params = self.mapping.count_parameters()
        node_params = total - mapping_params
        
        return {
            'total': total,
            'mapping': mapping_params,
            'nodes': node_params
        }
    
    def get_parameter_efficiency(self):
        """
        Calculate parameter efficiency compared to learnable layer.
        
        Returns:
            Dictionary with efficiency metrics
        """
        params = self.count_parameters()
        
        # LearnableLayer has a weight matrix of size (output_size * n, input_size)
        learnable_params = self.output_size * self.n * self.input_size
        reduction = 1.0 - (params['mapping'] / learnable_params)
        
        return {
            'grouped_params': params,
            'learnable_params': learnable_params,
            'reduction': reduction,
            'efficiency': learnable_params / params['total'] if params['total'] > 0 else 0
        }
    
    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return f"input_size={self.input_size}, output_size={self.output_size}, " \
               f"n={self.n}, num_groups={self.num_groups}, tau={self.tau}, mapping=grouped"
