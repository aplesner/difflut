import time

import torch
import torch.nn as nn

from difflut.layers.random_layer import RandomLayer
from difflut.nodes.dwn_node import DWNNode

def print_memory_stats(label: str, baseline_mb: float = 0.0):
    """Print current and peak memory statistics."""
    if torch.cuda.is_available():
        current = torch.cuda.memory_allocated() / (1024**2)
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        delta = current - baseline_mb
        print(f"[{label}] Current: {current:.2f} MB (Δ{delta:+.2f} MB) | Peak: {peak:.2f} MB | Reserved: {reserved:.2f} MB")
        return current
    return 0.0

class ConvolutionalLUTLayer(nn.Module):
    """
    Convolutional layer using LUT-based nodes with memory-efficient fused kernels.

    Uses RandomLayer with fused forward_with_mapping() to avoid materializing
    large intermediate tensors, reducing memory usage from ~20GB to ~3-4GB.

    Processes trees in chunks to balance speed (fewer kernel calls) and memory
    (not materializing all trees' activations at once).
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
            seed: int = 42,
            chunk_size: int = 32,
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
        self.seed = seed
        self.chunk_size = min(chunk_size, out_channels)  # Don't exceed out_channels

        # Build tree architecture
        hidden_layers = [self.n_inputs_per_node ** (self.tree_depth - i) for i in range(self.tree_depth + 1)]
        self.hidden_layers = hidden_layers

        # OPTIMIZATION: Create first layers in chunks for memory/speed balance
        # Processing chunk_size trees at once balances:
        # - Speed: chunk_size kernel calls instead of out_channels (e.g., 4 vs 128)
        # - Memory: Only materialize chunk_size trees' activations at once
        self.first_layer_chunks = nn.ModuleList()
        num_chunks = (out_channels + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, out_channels)
            actual_chunk_size = chunk_end - chunk_start

            if layer_type == 'random':
                chunk_layer = RandomLayer(
                    input_size=self.input_size,
                    output_size=hidden_layers[1] * actual_chunk_size,
                    node_type=DWNNode,
                    node_kwargs={'input_dim': n_inputs_per_node, 'output_dim': 1},
                    seed=seed + chunk_idx * 10000
                )
            else:
                raise ValueError(f"Unsupported layer_type: {layer_type}")

            self.first_layer_chunks.append(chunk_layer)

        # Create remaining layers per-tree (after first layer)
        # These remain independent as each tree has different activations after first layer
        self.trees = nn.ModuleList()

        for tree_idx in range(out_channels):
            # Build layers for this tree (starting from second layer)
            tree_layers = nn.ModuleList()
            current_input_size = hidden_layers[1]  # Output of first layer

            for layer_idx, output_size in enumerate(hidden_layers[2:]):  # Skip first two elements
                if layer_type == 'random':
                    layer = RandomLayer(
                        input_size=current_input_size,
                        output_size=output_size,
                        node_type=DWNNode,
                        node_kwargs={'input_dim': n_inputs_per_node, 'output_dim': 1},
                        seed=seed + tree_idx * 1000 + layer_idx + 1  # Unique seed per tree+layer
                    )
                else:
                    raise ValueError(f"Unsupported layer_type: {layer_type}")

                tree_layers.append(layer)
                current_input_size = output_size

            self.trees.append(tree_layers)

        # For convolution, we use the unfold operation
        self.unfold = nn.Unfold(kernel_size=receptive_field, padding=0, stride=1)

    def _pair(self, x: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(x, int):
            return (x, x)
        return x

    def __repr__(self) -> str:
        """Simple model overview without printing all trees."""
        num_chunks = len(self.first_layer_chunks)
        return (
            f"ConvolutionalLUTLayer(\n"
            f"  receptive_field={self.receptive_field}, stride={self.stride}, padding={self.padding}\n"
            f"  in_channels={self.in_channels}, out_channels={self.out_channels}\n"
            f"  tree_architecture={self.hidden_layers}\n"
            f"  node_type={self.node_type}, layer_type={self.layer_type}\n"
            f"  n_inputs_per_node={self.n_inputs_per_node}, tree_depth={self.tree_depth}\n"
            f"  total_trees={len(self.trees)}, chunk_size={self.chunk_size}, num_chunks={num_chunks}\n"
            f")"
        )

    def forward(self, x):
        from torch.profiler import record_function

        batch_size = x.shape[0]

        # Memory checkpoint: baseline
        baseline = print_memory_stats("Forward start (input ready)")

        # Extract patches: (batch, patch_size, num_patches)
        with record_function("unfold_operation"):
            patches = self.unfold(x)
            num_patches = patches.shape[2]

            # Reshape to (batch*num_patches, patch_size)
            patches = patches.transpose(1, 2).contiguous()
            patches = patches.view(-1, self.input_size)

            baseline = print_memory_stats(f"After unfold (patches: {patches.shape})", baseline)

        # Process patches through trees using chunked first layer + per-tree remaining layers
        with record_function("tree_processing"):
            batch_patches = patches.shape[0]

            # OPTIMIZATION: Preallocate output tensor
            output = torch.empty(batch_patches, self.out_channels, device=patches.device)

            # Process trees in chunks for memory/speed balance
            tree_idx = 0
            for chunk_idx, chunk_layer in enumerate(self.first_layer_chunks):
                with record_function(f"chunk_{chunk_idx}_first_layer"):
                    # Process this chunk's first layer
                    x_chunk = chunk_layer(patches)  # (batch*num_patches, hidden_layers[1] * chunk_size)
                    x_chunk = torch.clamp(x_chunk, 0, 1)

                    # Determine actual chunk size (last chunk may be smaller)
                    chunk_start = chunk_idx * self.chunk_size
                    chunk_end = min(chunk_start + self.chunk_size, self.out_channels)
                    actual_chunk_size = chunk_end - chunk_start

                    # Reshape to split per-tree: (batch*num_patches, chunk_size, hidden_layers[1])
                    x_chunk = x_chunk.view(batch_patches, actual_chunk_size, self.hidden_layers[1])

                # Process remaining layers for trees in this chunk
                for local_tree_idx in range(actual_chunk_size):
                    with record_function(f"tree_{tree_idx}_remaining_layers"):
                        # Get this tree's first layer output: (batch*num_patches, hidden_layers[1])
                        x_tree = x_chunk[:, local_tree_idx, :]

                        # Process through remaining layers of this tree
                        for layer in self.trees[tree_idx]:  # type: ignore
                            x_tree = layer(x_tree)
                            x_tree = torch.clamp(x_tree, 0, 1)

                        # Write directly to preallocated output
                        output[:, tree_idx] = x_tree.squeeze(-1)

                    tree_idx += 1

                # Print memory after each chunk
                baseline = print_memory_stats(f"After chunk {chunk_idx+1}/{len(self.first_layer_chunks)} ({tree_idx}/{self.out_channels} trees)", baseline)

            baseline = print_memory_stats(f"After all {self.out_channels} trees (output ready)", baseline)

        with record_function("reshape_output"):
            output = output.view(batch_size, num_patches, self.out_channels)
            output = output.transpose(1, 2)  # (batch, out_channels, num_patches)

            # Calculate output spatial dimensions
            out_h = (x.shape[2] + 2 * self.padding[0] - self.receptive_field[0]) // self.stride[0] + 1
            out_w = (x.shape[3] + 2 * self.padding[1] - self.receptive_field[1]) // self.stride[1] + 1

            output = output.view(batch_size, self.out_channels, out_h, out_w)

            baseline = print_memory_stats("After reshape (forward complete)", baseline)

        return output
        

batch_size = 128
image_size = 64
in_channels = 32
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
    chunk_size=32,  # Process 32 trees at a time (balance speed/memory)
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

