import time

import torch
import torch.nn as nn

from difflut.layers import ConvolutionalLayer, ConvolutionConfig, LayerConfig, RandomLayer
from difflut.nodes import DWNNode
from difflut.nodes.node_config import NodeConfig

batch_size = 128
image_size = 64
in_channels = 32
out_channels = 128

# Create convolution configuration
conv_config = ConvolutionConfig(
    tree_depth=2,
    in_channels=in_channels,
    out_channels=out_channels,
    receptive_field=5,
    stride=1,
    padding=0,
    chunk_size=32,  # Process 32 trees at a time (balance speed/memory)
    seed=42
)

# Create node configuration
node_config = NodeConfig(input_dim=6, output_dim=1)

# Create layer configuration for training (with bit flip and gradient stabilization)
layer_config = LayerConfig(
    flip_probability=0.05,  # 5% bit flipping for robustness
    grad_stabilization='layerwise',
    grad_target_std=1.0
)

# Create convolutional LUT layer
conv_lut_layer = ConvolutionalLayer(
    convolution_config=conv_config,
    node_type=DWNNode,
    node_kwargs=node_config,
    layer_type=RandomLayer,
    n_inputs_per_node=6,
    layer_config=layer_config  # Use LayerConfig for training parameters
)

# Print model summary
print(conv_lut_layer)

# get some random binary input data
input_data = torch.randint(0, 2, (batch_size, in_channels, image_size, image_size)).float()

if torch.cuda.is_available():
    conv_lut_layer = conv_lut_layer.cuda()
    input_data = input_data.cuda()


# from torch.profiler import profile, ProfilerActivity, record_function

# activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    device = "cuda"
    # activities += [ProfilerActivity.CUDA]

# with profile(
#         activities=activities, profile_memory=True, record_shapes=True
    # ) as prof:
start_time = time.time()
with torch.no_grad():
    output = conv_lut_layer(input_data)
end_time = time.time()

# print("\n" + "="*100)
# print("GPU Memory Usage by Operation")
# print("="*100)
# print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

# print("\n" + "="*100)
# print("Time and Memory by Operation")
# print("="*100)
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("\n" + "="*100)
print("Peak Memory Usage")
print("="*100)
if torch.cuda.is_available():
    print(f"Peak CUDA memory allocated: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
    print(f"Peak CUDA memory reserved: {torch.cuda.max_memory_reserved() / (1024**2):.2f} MB")
else:
    print("CPU profiling - peak memory tracking not available via torch.cuda")

print("\n" + "="*100)
print("Input shape:", input_data.shape  )
print("Output shape:", output.shape)
print(f"Forward pass took {end_time - start_time:.2f} seconds")

