import time

import torch
import torch.nn as nn

from difflut.models.feedforward import feedforward_core

class ConvolutionalLUTLayer(nn.Module):
    """
    Convolutional layer using LUT-based nodes
    """
    
    def __init__(
            self,
            tree_depth: int,
            in_channels: int,
            out_channels: int,
            receptive_field: int | tuple[int, int] = 5,
            stride: int | tuple[int, int] = 1,
            padding: int | tuple[int, int] = 0,
            node_type: str = 'dwn',
            layer_type: str = 'random',
            n_inputs_per_node: int = 6,
            # node_kwargs: dict | None = None
            ):
        super(ConvolutionalLUTLayer, self).__init__()
        self.tree_depth = tree_depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.receptive_field = self._pair(receptive_field)
        self.input_size = in_channels * self.receptive_field[0] * self.receptive_field[1]
        self.stride = self._pair(stride)
        self.padding = self._pair(padding)
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
                input_size=self.input_size,
                hidden_sizes=hidden_layers,
                node_type=self.node_type,
                layer_type=self.layer_type,
                n_inputs=self.n_inputs_per_node,
                # node_kwargs=self.node_kwargs
            )
            self.trees.append(tree)
        
        # For convolution, we use the unfold operation
        self.unfold = nn.Unfold(kernel_size=receptive_field, padding=0, stride=1)

    def _pair(self, x: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(x, int):
            return (x, x)
        return x

    def forward(self, x):

        batch_size = x.shape[0]

        # Extract patches: (batch, patch_size, num_patches)
        patches = self.unfold(x)
        num_patches = patches.shape[2]
        
        # Reshape to (batch*num_patches, patch_size)
        patches = patches.transpose(1, 2).contiguous()
        patches = patches.view(-1, self.input_size)

        # Process each patch through each tree
        output = [tree(patches) for tree in self.trees]
        output = torch.stack(output, dim=1)  # (batch*num_patches, out_channels)

        output = output.view(batch_size, num_patches, self.out_channels)
        output = output.transpose(1, 2)  # (batch, out_channels, num_patches)
        
        # Calculate output spatial dimensions
        out_h = (x.shape[2] + 2 * self.padding[0] - self.receptive_field[0]) // self.stride[0] + 1
        out_w = (x.shape[3] + 2 * self.padding[1] - self.receptive_field[1]) // self.stride[1] + 1

        output = output.view(batch_size, self.out_channels, out_h, out_w)
        
        return output
        

batch_size = 128
image_size = 28
in_channels = 1
out_channels = 128

conv_lut_layer = ConvolutionalLUTLayer(
    tree_depth=2,
    in_channels=in_channels,
    out_channels=out_channels,
    receptive_field=5,
    stride=1,
    padding=0,
    node_type='dwn',
    layer_type='random',
    n_inputs_per_node=6,
)
    

# get some random binary input data
input_data = torch.randint(0, 2, (batch_size, 1, image_size, image_size)).float()

if torch.cuda.is_available():
    conv_lut_layer = conv_lut_layer.cuda()
    input_data = input_data.cuda()


from torch.profiler import profile, ProfilerActivity, record_function

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    device = "cuda"
    activities += [ProfilerActivity.CUDA]

with profile(
        activities=activities, profile_memory=True, record_shapes=True
    ) as prof:
    start_time = time.time()
    output = conv_lut_layer(input_data)
    end_time = time.time()

print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

print("Output shape:", output.shape)
print(f"Forward pass took {end_time - start_time:.2f} seconds")

