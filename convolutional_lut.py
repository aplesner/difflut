import time

import torch
import torch.nn as nn

from difflut.models.feedforward import feedforward_core

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
        self.hidden_layers = hidden_layers  # Store for memory diagnostics
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

        # Print LUT parameter memory diagnostics
        # self._print_lut_memory_info()

    def _pair(self, x: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(x, int):
            return (x, x)
        return x

    def _print_lut_memory_info(self):
        """Print LUT parameter memory information."""
        print("\n" + "="*80)
        print("LUT PARAMETER MEMORY ANALYSIS")
        print("="*80)
        print(f"Architecture: {self.input_size} → {self.hidden_layers}")
        print(f"Output channels (trees): {self.out_channels}")
        print(f"Inputs per node: {self.n_inputs_per_node}")
        print(f"LUT entries per node: 2^{self.n_inputs_per_node} = {2**self.n_inputs_per_node}")

        total_luts = 0
        for i, layer_size in enumerate(self.hidden_layers):
            lut_entries = layer_size * (2 ** self.n_inputs_per_node)
            memory_mb = lut_entries * 4 / (1024**2)  # 4 bytes per float32
            total_luts += lut_entries
            print(f"  Layer {i+1}: {layer_size} nodes × {2**self.n_inputs_per_node} entries = {lut_entries} LUTs ({memory_mb:.2f} MB)")

        per_tree_luts = total_luts
        per_tree_mb = per_tree_luts * 4 / (1024**2)
        total_luts_all_trees = per_tree_luts * self.out_channels
        total_mb = total_luts_all_trees * 4 / (1024**2)

        print(f"\nPer tree: {per_tree_luts} LUT entries ({per_tree_mb:.2f} MB)")
        print(f"Total ({self.out_channels} trees): {total_luts_all_trees} LUT entries ({total_mb:.2f} MB)")
        print("="*80 + "\n")

    def _print_activation_memory_estimates(self, num_patches: int):
        """Print theoretical activation memory estimates for tree processing."""
        print("\n" + "="*80)
        print("THEORETICAL ACTIVATION MEMORY ESTIMATES")
        print("="*80)
        print(f"Number of patches: {num_patches}")
        print(f"Number of trees: {self.out_channels}")
        print(f"\nPer-tree activation memory:")

        total_per_tree_mb = 0
        for i, layer_size in enumerate(self.hidden_layers):
            activation_elements = num_patches * layer_size
            memory_mb = activation_elements * 4 / (1024**2)  # 4 bytes per float32
            total_per_tree_mb += memory_mb
            print(f"  Layer {i+1}: {num_patches} patches × {layer_size} nodes = {activation_elements} values ({memory_mb:.2f} MB)")

        print(f"\nTotal per tree (forward): {total_per_tree_mb:.2f} MB")
        print(f"Total per tree (with gradients): {total_per_tree_mb * 2:.2f} MB")

        # Sequential processing (current implementation)
        output_per_tree = num_patches * 1  # Each tree outputs 1 value per patch
        output_memory_mb = output_per_tree * 4 / (1024**2)
        all_outputs_mb = output_memory_mb * self.out_channels
        print(f"\nTree outputs (stored in list):")
        print(f"  Per tree output: {num_patches} patches × 1 output = {output_per_tree} values ({output_memory_mb:.2f} MB)")
        print(f"  All {self.out_channels} trees: {all_outputs_mb:.2f} MB")

        # Worst case: one tree + all outputs stored
        worst_case_mb = total_per_tree_mb * 2 + all_outputs_mb  # *2 for gradients
        print(f"\nWorst-case peak (1 tree forward+backward + all outputs): {worst_case_mb:.2f} MB")

        # Best case estimate: intermediate activations for all trees
        all_trees_activation_mb = total_per_tree_mb * self.out_channels
        all_trees_with_grad_mb = all_trees_activation_mb * 2
        print(f"If all tree activations kept in memory: {all_trees_with_grad_mb:.2f} MB")

        print("="*80 + "\n")

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

            after_unfold = print_memory_stats(f"After unfold (patches: {patches.shape})", baseline)

        # Process each patch through each tree
        with record_function("tree_processing"):
            before_trees = print_memory_stats("Before tree processing", baseline)

            # Print activation memory estimates
            # self._print_activation_memory_estimates(num_patches*batch_size)

            output_list = []
            for i, tree in enumerate(self.trees):
                with record_function(f"tree_{i}_forward"):
                    tree_output = tree(patches)
                    output_list.append(tree_output)

                # Print every 32 trees
                if (i + 1) % 32 == 0:
                    baseline = print_memory_stats(f"After {i+1}/{self.out_channels} trees", baseline)

            after_trees = print_memory_stats(f"After all {self.out_channels} trees (before stack)", baseline)

            output = torch.stack(output_list, dim=1)  # (batch*num_patches, out_channels)

            after_stack = print_memory_stats("After torch.stack", after_trees)

        with record_function("reshape_output"):
            output = output.view(batch_size, num_patches, self.out_channels)
            output = output.transpose(1, 2)  # (batch, out_channels, num_patches)

            # Calculate output spatial dimensions
            out_h = (x.shape[2] + 2 * self.padding[0] - self.receptive_field[0]) // self.stride[0] + 1
            out_w = (x.shape[3] + 2 * self.padding[1] - self.receptive_field[1]) // self.stride[1] + 1

            output = output.view(batch_size, self.out_channels, out_h, out_w)

            print_memory_stats("After reshape (forward complete)", baseline)

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

print("\n" + "="*100)
print("GPU Memory Usage by Operation")
print("="*100)
print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

print("\n" + "="*100)
print("Time and Memory by Operation")
print("="*100)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("\n" + "="*100)
print("Peak Memory Usage")
print("="*100)
if torch.cuda.is_available():
    print(f"Peak CUDA memory allocated: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
    print(f"Peak CUDA memory reserved: {torch.cuda.max_memory_reserved() / (1024**2):.2f} MB")
else:
    print("CPU profiling - peak memory tracking not available via torch.cuda")

print("Output shape:", output.shape)
print(f"Forward pass took {end_time - start_time:.2f} seconds")

