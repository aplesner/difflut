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
    
    Optimizations:
    - Vectorized operations instead of loops where possible
    - Efficient gather operations in eval mode
    - Better weight initialization for faster convergence
    """
    
    def __init__(self, 
                 input_size: int, 
                 output_size: int,
                 n_inputs_per_node: int,
                 num_groups: int = 4,
                 tau: float = 0.001,
                 overlap: float = 0.0):
        """
        Args:
            input_size: Total number of input features
            output_size: Number of nodes (each node needs n connections)
            n_inputs_per_node: Number of inputs each node needs (n)
            num_groups: Number of groups to divide the input space into
            tau: Temperature for softmax during training
            overlap: Fraction of overlap between adjacent groups (0.0-0.5) for better accuracy
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_inputs_per_node = n_inputs_per_node
        self.num_groups = num_groups
        self.tau = tau
        self.overlap = overlap
        
        # Calculate group size and nodes per group
        base_group_size = input_size // num_groups
        overlap_size = int(base_group_size * overlap)
        self.group_size = base_group_size + overlap_size
        self.nodes_per_group = (output_size + num_groups - 1) // num_groups
        
        # Create weight matrices for each group
        # Each group has nodes_per_group nodes, each selecting from group_size inputs
        self.group_weights = nn.ParameterList([
            nn.Parameter(torch.randn(
                min(self.nodes_per_group, output_size - i * self.nodes_per_group),
                n_inputs_per_node,
                min(self.group_size, input_size - max(0, i * base_group_size - overlap_size))
            ))
            for i in range(num_groups)
        ])
        
        # Better initialization: uniform distribution with small positive bias
        # This encourages initial exploration of all inputs
        for weight in self.group_weights:
            nn.init.xavier_uniform_(weight)
            # Add small positive bias to prevent initial collapse to single input
            with torch.no_grad():
                weight.add_(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with grouped connections (optimized vectorized version).
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        Returns:
            Output tensor of shape (batch_size, output_size, n_inputs_per_node)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Calculate base group size for overlap handling
        base_group_size = self.input_size // self.num_groups
        overlap_size = int(base_group_size * self.overlap)
        
        if self.training:
            # TRAINING MODE: Soft selection with vectorized operations
            # Pre-allocate output tensor
            output = torch.zeros(batch_size, self.output_size, self.n_inputs_per_node, 
                               device=device, dtype=x.dtype)
            
            node_offset = 0
            
            for group_idx in range(self.num_groups):
                group_nodes = min(self.nodes_per_group, self.output_size - node_offset)
                
                if group_nodes == 0:
                    break
                
                # Calculate input range with overlap
                input_start = max(0, group_idx * base_group_size - overlap_size)
                input_end = min(self.input_size, input_start + self.group_size)
                group_inputs = input_end - input_start
                
                # Get the input slice for this group
                x_group = x[:, input_start:input_end]
                
                # Get weights and compute soft selection
                W = self.group_weights[group_idx]
                weights = F.softmax(W / self.tau, dim=-1)
                
                # Efficient einsum operation
                output[:, node_offset:node_offset + group_nodes, :] = \
                    torch.einsum('bi,oni->bon', x_group, weights)
                
                node_offset += group_nodes
        else:
            # EVALUATION MODE: Hard selection with fully vectorized gather
            # Build a global index tensor for all groups at once
            global_indices = torch.zeros(self.output_size, self.n_inputs_per_node, 
                                        dtype=torch.long, device=device)
            
            node_offset = 0
            
            for group_idx in range(self.num_groups):
                group_nodes = min(self.nodes_per_group, self.output_size - node_offset)
                
                if group_nodes == 0:
                    break
                
                # Calculate input range with overlap
                input_start = max(0, group_idx * base_group_size - overlap_size)
                
                # Get hard indices for this group
                W = self.group_weights[group_idx]
                hard_indices = torch.argmax(W, dim=-1)  # (group_nodes, n)
                
                # Add input offset to convert to global indices
                global_indices[node_offset:node_offset + group_nodes] = \
                    hard_indices + input_start
                
                node_offset += group_nodes
            
            # Single vectorized gather operation for all nodes
            # Expand x for gathering: (batch_size, output_size, input_size)
            x_expanded = x.unsqueeze(1).expand(batch_size, self.output_size, self.input_size)
            # Expand indices: (batch_size, output_size, n)
            indices_expanded = global_indices.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Gather all at once
            output = torch.gather(x_expanded, 2, indices_expanded)
        
        return output
    
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
                 tau: float = 0.001,
                 overlap: float = 0.0):
        """
        Args:
            input_size: Size of input vector
            output_size: Number of LUT nodes
            node_type: LUT node class
            n: Number of inputs per LUT
            node_kwargs: Additional arguments for nodes
            num_groups: Number of groups to divide inputs and nodes into
            tau: Temperature for softmax in grouped mapping
            overlap: Overlap fraction between groups (0.0-0.5) for better accuracy
        """
        # Initialize parent with nodes
        super().__init__(input_size, output_size, node_type, n, node_kwargs)
        
        self.num_groups = num_groups
        self.tau = tau
        self.overlap = overlap
        
        # Create grouped mapping module
        self.mapping = GroupedMappingModule(
            input_size=input_size,
            output_size=output_size,
            n_inputs_per_node=n,
            num_groups=num_groups,
            tau=tau,
            overlap=overlap
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
            
            # Calculate base group size for overlap handling
            base_group_size = self.input_size // self.num_groups
            overlap_size = int(base_group_size * self.overlap)
            
            node_offset = 0
            
            for group_idx in range(self.num_groups):
                group_nodes = min(self.mapping.nodes_per_group, 
                                self.output_size - node_offset)
                
                if group_nodes == 0:
                    break
                
                # Calculate input range with overlap
                input_start = max(0, group_idx * base_group_size - overlap_size)
                
                W = self.mapping.group_weights[group_idx]
                hard_indices = torch.argmax(W, dim=-1)  # (group_nodes, n)
                
                # Add input offset to get global indices
                mapping_matrix[node_offset:node_offset + group_nodes] = \
                    hard_indices + input_start
                
                node_offset += group_nodes
            
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
               f"n={self.n}, num_groups={self.num_groups}, tau={self.tau}, " \
               f"overlap={self.overlap}, mapping=grouped"
