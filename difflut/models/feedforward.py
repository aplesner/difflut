from typing import Type

import torch
import torch.nn as nn

from ..utils.modules import GroupSum
from ..registry import REGISTRY
from ..layers.base_layer import BaseLUTLayer
from ..nodes.base_node import BaseNode

class feedforward_core(nn.Module):
    """
    Core DiffLUT feedforward model without encoder and output layer.
    
    Built using:
    - Hidden layers: LUT-based layers with configurable node type
    """
    
    def __init__(
            self, 
            input_size: int, 
            hidden_sizes: list[int] = [1000, 1000], 
            node_type: str = 'dwn', 
            layer_type: str = 'random', 
            n_inputs: int = 6
        ):
        """
        Args:
            input_size: Size of input
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            node_type: Type of LUT node to use (from registry)
            layer_type: Type of layer connection to use (from registry)
            n_inputs: Number of inputs per LUT node
        """
        super().__init__()
        
        self.input_size = input_size

        # Get node and layer classes from registry
        node_class: Type[BaseNode] = REGISTRY.get_node(node_type)
        layer_class: Type[BaseLUTLayer] = REGISTRY.get_layer(layer_type)
        
        # print(f"\nBuilding DiffLUT model:")
        # print(f"  Node type: {node_type}")
        # print(f"  Input size: {input_size}")
        # print(f"  Hidden layers: {hidden_sizes}")
        
        # Build hidden layers
        self.hidden_layers = nn.ModuleList()
        current_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            node_kwargs = {
                'input_dim': n_inputs,
            }
            
            # Create layer with proper node_kwargs
            layer = layer_class(
                input_size=current_size,
                output_size=hidden_size,
                node_type=node_class,
                node_kwargs=node_kwargs,
            )
            self.hidden_layers.append(layer)
            # print(f"  Layer {i+1}: {current_size} → {hidden_size}")
            current_size = hidden_size

        self.output_size = current_size

    def forward(self, x):
        """Forward pass through encoder and all layers."""
        # Encode input
        x = torch.clamp(x, 0, 1)  # Ensure binary range
        
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.clamp(x, 0, 1)  # Keep outputs in valid range
        
        return x


class feedforward(nn.Module):
    """
    Simple DiffLUT feedforward model with encoder and output layer.
    
    Built using:
    - Encoder: Converts continuous values to binary (e.g., Thermometer encoding)
    - Hidden layers: LUT-based layers with configurable node type
    - Output layer: GroupSum to reduce nodes to class logits
    """
    
    def __init__(
            self, 
            encoder, 
            encoded_size: int, 
            hidden_sizes: list[int] = [1000, 1000], 
            num_classes: int = 10, 
            node_type: str = 'dwn', 
            layer_type: str = 'random', 
            n_inputs: int = 6
        ):
        """
        Args:
            encoder: Fitted encoder (e.g., ThermometerEncoder)
            encoded_size: Size of encoded input (after encoder)
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            node_type: Type of LUT node to use (from registry)
            n_inputs: Number of inputs per LUT node
        """
        super().__init__()
        
        self.encoder = encoder
        self.encoded_size = encoded_size

        self.core = feedforward_core(
            input_size=encoded_size,
            hidden_sizes=hidden_sizes,
            node_type=node_type,
            layer_type=layer_type,
            n_inputs=n_inputs 
        )


        # Output layer: GroupSum to reduce to num_classes
        if self.core.output_size % num_classes != 0:
            # If not divisible, it's okay - GroupSum will handle it
            print(f"  Output: {self.core.output_size} → {num_classes} (via GroupSum)")
        else:
            print(f"  Output: {self.core.output_size} → {num_classes} (via GroupSum, {self.core.output_size//num_classes} per class)")

        self.output_layer = GroupSum(k=num_classes, tau=1)
    
    def forward(self, x):
        """Forward pass through encoder and all layers."""
        # Encode input
        x = self.encoder.encode(x)

        x = self.core(x)

        # Group and sum for classification
        x = self.output_layer(x)
        return x