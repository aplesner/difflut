# Convolutional kernel for LUT-based models
from typing import Optional, Type

import torch
import torch.nn as nn

from ..nodes.node_config import NodeConfig
from ..registry import register_convolutional_layer
from .base_layer import LUTLayerMixin
from .layer_config import LayerConfig, GroupedInputConfig


class ConvolutionConfig:
    """
    Configuration for ConvolutionalLUTLayer.
    """

    def __init__(
        self,
        tree_depth: int = 2,
        in_channels: int | None = None,
        out_channels: int | None = None,
        receptive_field: int | tuple[int, int] = 5,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        chunk_size: int = 32,
        seed: int = 42,
    ):
        assert in_channels is not None, "in_channels must be specified"
        assert out_channels is not None, "out_channels must be specified"

        self.tree_depth = tree_depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.receptive_field = receptive_field
        self.stride = stride
        self.padding = padding
        self.chunk_size = chunk_size
        self.seed = seed


@register_convolutional_layer("convolutional")
class ConvolutionalLayer(LUTLayerMixin, nn.Module):
    """
    Convolutional layer using LUT-based nodes with memory-efficient fused kernels.

    Uses RandomLayer with fused forward_with_mapping() to avoid materializing
    large intermediate tensors, reducing memory usage from ~20GB to ~3-4GB.

    Processes trees in chunks to balance speed (fewer kernel calls) and memory
    (not materializing all trees' activations at once).
    """

    def __init__(
        self,
        convolution_config: ConvolutionConfig,
        node_type: Type[nn.Module],
        node_kwargs: NodeConfig,
        layer_type: Type[nn.Module],
        n_inputs_per_node: int = 6,
        layer_config: Optional[LayerConfig] = None,
        flip_probability: Optional[float] = None,
        grad_stabilization: Optional[str] = None,
        grad_target_std: Optional[float] = None,
        grad_subtract_mean: Optional[bool] = None,
        grad_epsilon: Optional[float] = None,
        grouped_connections: bool = False,
        ensure_full_coverage: bool = False,
    ):
        super().__init__()

        # Initialize LUTLayerMixin for bit flip and gradient stabilization
        self._init_lut_layer_mixin(
            layer_config,
            flip_probability,
            grad_stabilization,
            grad_target_std,
            grad_subtract_mean,
            grad_epsilon,
        )

        self.tree_depth = convolution_config.tree_depth
        self.in_channels = convolution_config.in_channels
        self.out_channels = convolution_config.out_channels
        self.receptive_field = self._pair(convolution_config.receptive_field)
        self.input_size = self.in_channels * self.receptive_field[0] * self.receptive_field[1]
        self.stride = self._pair(convolution_config.stride)
        self.padding = self._pair(convolution_config.padding)
        self.node_type = node_type
        self.layer_type = layer_type
        self.n_inputs_per_node = n_inputs_per_node
        self.seed = convolution_config.seed
        self.chunk_size = min(
            convolution_config.chunk_size, self.out_channels
        )  # Don't exceed out_channels

        # Build tree architecture
        hidden_layers = [
            self.n_inputs_per_node ** (self.tree_depth - i) for i in range(self.tree_depth + 1)
        ]
        self.hidden_layers = hidden_layers

        # Create grouped input mapping if enabled
        if grouped_connections:
            n_groups = self.in_channels
            assert (
                self.input_size % n_groups == 0
            ), "Input size must be divisible by number of groups. If you see this then somehthing has gone very wrong..."

            grouped_config = GroupedInputConfig(
                n_groups=self.in_channels,
                input_size=self.input_size,
                output_trees=self.out_channels,
                luts_per_tree=hidden_layers[0],
                bits_per_node=self.n_inputs_per_node,
                seed=self.seed,
                ensure_full_coverage=ensure_full_coverage,
            )
        else:
            grouped_config = None

        # OPTIMIZATION: Create first layers in chunks for memory/speed balance
        # Processing chunk_size trees at once balances:
        # - Speed: chunk_size kernel calls instead of out_channels (e.g., 4 vs 128)
        # - Memory: Only materialize chunk_size trees' activations at once
        self.first_layer_chunks = nn.ModuleList()
        num_chunks = (self.out_channels + self.chunk_size - 1) // self.chunk_size

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, self.out_channels)
            actual_chunk_size = chunk_end - chunk_start

            layer = layer_type(
                input_size=self.input_size,
                output_size=hidden_layers[0] * actual_chunk_size,
                node_type=node_type,
                node_kwargs=node_kwargs,
                mapping_indices=grouped_config.get_mapping_indices(chunk_start, chunk_end) if grouped_config else None,
            )

            self.first_layer_chunks.append(layer)

        # Create remaining layers per-tree (after first layer)
        # These remain independent as each tree has different activations after first layer
        self.trees = nn.ModuleList()

        for tree_idx in range(self.out_channels):
            # Build layers for this tree (starting from second layer)
            tree_layers = nn.ModuleList()
            current_input_size = hidden_layers[0]  # Output of first layer

            for layer_idx, output_size in enumerate(hidden_layers[1:]):  # Skip first two elements
                layer = layer_type(
                    input_size=current_input_size,
                    output_size=output_size,
                    node_type=node_type,
                    node_kwargs=node_kwargs,
                )

                tree_layers.append(layer)
                current_input_size = output_size

            self.trees.append(tree_layers)

        # For convolution, we use the unfold operation to extract patches
        self.unfold = nn.Unfold(
            kernel_size=self.receptive_field, padding=self.padding, stride=self.stride
        )

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

        batch_size = x.shape[0]

        # Extract patches: (batch, patch_size, num_patches)
        patches = self.unfold(x)
        num_patches = patches.shape[2]

        # Reshape to (batch*num_patches, patch_size)
        patches = patches.transpose(1, 2).contiguous()
        patches = patches.view(-1, self.input_size)
        # Apply bit-flip augmentation during training
        if self.training and self.flip_probability > 0.0:
            patches = self._apply_bit_flip(patches)

        # Process patches through trees using chunked first layer + per-tree remaining layers
        batch_patches = patches.shape[0]

        # OPTIMIZATION: Preallocate output tensor
        output = torch.empty(batch_patches, self.out_channels, device=patches.device)

        # Process trees in chunks for memory/speed balance
        tree_idx = 0
        for chunk_idx, chunk_layer in enumerate(self.first_layer_chunks):
            # Process this chunk's first layer
            x_chunk = chunk_layer(patches)  # (batch*num_patches, hidden_layers[0] * chunk_size)

            # Determine actual chunk size (last chunk may be smaller)
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, self.out_channels)
            actual_chunk_size = chunk_end - chunk_start

            # Reshape to split per-tree: (batch*num_patches, chunk_size, hidden_layers[1])
            x_chunk = x_chunk.view(batch_patches, actual_chunk_size, self.hidden_layers[0])

            # Process remaining layers for trees in this chunk
            for local_tree_idx in range(actual_chunk_size):
                # Get this tree's first layer output: (batch*num_patches, hidden_layers[1])
                x_tree = x_chunk[:, local_tree_idx, :]

                # Process through remaining layers of this tree
                for layer in self.trees[tree_idx]:  # type: ignore
                    x_tree = layer(x_tree)

                # Write directly to preallocated output
                output[:, tree_idx] = x_tree.squeeze(-1)

                tree_idx += 1

        output = output.view(batch_size, num_patches, self.out_channels)
        output = output.transpose(1, 2)  # (batch, out_channels, num_patches)

        # Calculate output spatial dimensions
        out_h = (x.shape[2] + 2 * self.padding[0] - self.receptive_field[0]) // self.stride[0] + 1
        out_w = (x.shape[3] + 2 * self.padding[1] - self.receptive_field[1]) // self.stride[1] + 1

        output = output.view(batch_size, self.out_channels, out_h, out_w)

        # Register gradient stabilization hook if enabled
        if self.grad_stabilization != "none" and self.training and output.requires_grad:
            # Flatten spatial dimensions for gradient stabilization
            # (batch, channels, h, w) -> (batch, channels*h*w)
            def grad_hook(grad):
                original_shape = grad.shape
                # Flatten to 2D: (batch, channels*h*w)
                grad_flat = grad.view(grad.shape[0], -1)
                # Apply stabilization
                grad_stabilized = self._apply_gradient_stabilization(grad_flat)
                # Reshape back to original
                return grad_stabilized.view(original_shape)

            output.register_hook(grad_hook)

        return output
