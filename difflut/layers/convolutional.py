

# Convolutional kernel for LUT-based models
import torch
import torch.nn as nn

from ..models.feedforward import feedforward_core

class ConvolutionalLUTLayer(nn.Module):
    """
    Convolutional layer using LUT-based nodes
    """
    
    def __init__(
            self,
            tree_depth: int,
            in_channels: int,
            out_channels: int,
            receptive_field: int,
            node_type: str = 'dwn',
            layer_type: str = 'random',
            n_inputs_per_node: int = 6,
            # node_kwargs: dict | None = None
            ):
        super(ConvolutionalLUTLayer, self).__init__()
        self.tree_depth = tree_depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.receptive_field = receptive_field
        self.node_type = node_type
        self.layer_type = layer_type
        self.n_inputs_per_node = n_inputs_per_node
        # self.node_kwargs = node_kwargs if node_kwargs is not None else {}

        # Create trees (one for each output channel)
        # Each tree is a small feedforward network of LUT nodes
        hidden_layers = [self.n_inputs_per_node ** (self.tree_depth - i) for i in range(self.tree_depth + 1)]
        self.trees = nn.ModuleList()
        for _ in range(out_channels):
            tree = feedforward_core(
                input_size=in_channels * receptive_field,
                hidden_sizes=hidden_layers,
                node_type=self.node_type,
                layer_type=self.layer_type,
                n_inputs=self.n_inputs_per_node,
                # node_kwargs=self.node_kwargs
            )
            self.trees.append(tree)
        
        # For convolution, we use the unfold operation
        self.unfold = nn.Unfold(kernel_size=receptive_field, padding=0, stride=1)

    def forward(self, x):

        batch_size = x.shape[0]
        
        # Extract patches: (batch, patch_size, num_patches)
        patches = self.unfold(x)
        num_patches = patches.shape[2]
        
        # Reshape to (batch*num_patches, patch_size)
        patches = patches.transpose(1, 2).contiguous()
        patches = patches.view(-1, self.receptive_field * self.in_channels)
        