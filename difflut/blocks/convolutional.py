# Convolutional block for LUT-based models
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn

from ..layers.base_layer import LUTLayerMixin
from ..layers.layer_config import GroupedInputConfig, LayerConfig
from ..nodes.node_config import NodeConfig
from ..registry import register_block
from .base_block import BaseLUTBlock
from .block_config import BlockConfig


@register_block("convolutional")
class ConvolutionalLayer(BaseLUTBlock, LUTLayerMixin):
    """
    Simplified convolutional block using LUT-based nodes.

    A composable block module that applies tree-based convolutional processing
    using LUT nodes. It reuses the same tree weights for all spatial positions
    (like standard convolution).

    Strategy:
    1. Build a tree of layers once (reusable across all spatial positions)
    2. Use nn.Unfold to extract patches from input image
    3. Process all patches through the same tree in parallel
    4. Reshape output back to spatial format

    This block can be used within larger models or as a standalone module.
    """

    def __init__(
        self,
        config: BlockConfig,
        node_type: Type[nn.Module],
        layer_type: Type[nn.Module],
    ) -> None:
        """
        Initialize convolutional block from BlockConfig.

        Parameters:
        - config: BlockConfig, Configuration for the convolutional block
        - node_type: Type[nn.Module], LUT node class to use
        - layer_type: Type[nn.Module], Layer class to use

        Required in config:
        - in_channels: int, Number of input channels
        - out_channels: int, Number of output channels
        - tree_depth: int, Depth of tree architecture (default: 2)

        Optional in config:
        - receptive_field: Tuple[int, int], Size of receptive field (default: (5, 5))
        - stride: Tuple[int, int], Stride for convolution (default: (1, 1))
        - padding: Tuple[int, int], Padding for convolution (default: (0, 0))
        - patch_chunk_size: Optional[int], Process patches in chunks to save memory
        - n_inputs_per_node: int, Number of inputs per node (default: 6)
        - flip_probability: float, Probability of flipping bits during training
        - grad_stabilization: str, Gradient stabilization mode
        - grad_target_std: float, Target standard deviation for gradients
        - grad_subtract_mean: bool, Subtract mean during gradient stabilization
        - grad_epsilon: float, Epsilon for numerical stability
        - grouped_connections: bool, Use channel-aware connections
        - ensure_full_coverage: bool, Ensure each input is used at least once
        """
        BaseLUTBlock.__init__(self, config)
        # Don't call LUTLayerMixin.__init__ directly; it will be called via _init_lut_layer_mixin

        # Validate required parameters
        if config.in_channels is None:
            raise ValueError("config.in_channels must be specified")
        if config.out_channels is None:
            raise ValueError("config.out_channels must be specified")

        # Set defaults for optional parameters
        tree_depth = config.tree_depth if config.tree_depth is not None else 2
        receptive_field = config.receptive_field if config.receptive_field is not None else (5, 5)
        stride = config.stride if config.stride is not None else (1, 1)
        padding = config.padding if config.padding is not None else (0, 0)
        n_inputs_per_node = config.n_inputs_per_node if config.n_inputs_per_node is not None else 6

        # Extract training parameters from config
        flip_probability = config.flip_probability
        grad_stabilization = config.grad_stabilization
        grad_target_std = config.grad_target_std
        grad_subtract_mean = config.grad_subtract_mean
        grad_epsilon = config.grad_epsilon
        grouped_connections = config.grouped_connections
        ensure_full_coverage = config.ensure_full_coverage

        # Create LayerConfig from block config for LUTLayerMixin
        layer_config = LayerConfig(
            flip_probability=flip_probability,
            grad_stabilization=grad_stabilization,
            grad_target_std=grad_target_std,
            grad_subtract_mean=grad_subtract_mean,
            grad_epsilon=grad_epsilon,
        )

        # Initialize LUTLayerMixin for bit flip and gradient stabilization
        self._init_lut_layer_mixin(layer_config)

        # Store block configuration
        self.tree_depth = tree_depth
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.receptive_field = self._pair(receptive_field)
        self.input_size = self.in_channels * self.receptive_field[0] * self.receptive_field[1]
        self.stride = self._pair(stride)
        self.padding = self._pair(padding)
        self.node_type = node_type
        self.layer_type = layer_type
        self.n_inputs_per_node = n_inputs_per_node
        self.seed = config.seed
        self.patch_chunk_size = config.patch_chunk_size

        # Build tree architecture: each layer reduces by factor of n_inputs_per_node
        # Example: tree_depth=2, n_inputs_per_node=6
        #   Layer 0: input_size -> 36 nodes (6^2)
        #   Layer 1: 36 -> 6 nodes (6^1)
        #   Layer 2: 6 -> 1 node (6^0) per output channel
        hidden_layers = [
            self.n_inputs_per_node ** (self.tree_depth - i) for i in range(self.tree_depth + 1)
        ]
        self.hidden_layers = hidden_layers

        # Create node configuration from block config
        node_kwargs = (
            NodeConfig(**config.node_kwargs)
            if config.node_kwargs
            else NodeConfig(
                input_dim=n_inputs_per_node,
                output_dim=1,
            )
        )

        # Create grouped input mapping if enabled (for channel-aware connections)
        if grouped_connections:
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

        # Build separate trees, one for each output channel
        # Each tree is reused for all spatial positions (weight sharing like conv)
        self.trees = nn.ModuleList()

        for tree_idx in range(self.out_channels):
            # Build one tree: input_size -> ... -> 1
            tree_layers = nn.ModuleList()

            # First layer: input_size -> hidden_layers[0]
            first_layer = layer_type(
                input_size=self.input_size,
                output_size=hidden_layers[0],
                node_type=node_type,
                node_kwargs=node_kwargs,
                mapping_indices=(
                    grouped_config.get_mapping_indices(tree_idx, tree_idx + 1)
                    if grouped_config
                    else None
                ),
            )
            tree_layers.append(first_layer)

            # Remaining layers: progressively reduce to 1 output
            current_input_size = hidden_layers[0]
            for layer_idx in range(1, len(hidden_layers)):
                output_size = hidden_layers[layer_idx]

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

    def _pair(self, x):
        # type: (int | Tuple[int, int]) -> Tuple[int, int]
        """Convert single int or tuple to (int, int) pair."""
        if isinstance(x, int):
            return (x, x)
        return x

    def get_config_summary(self) -> dict:
        """Get a summary of the block's configuration."""
        return {
            "block_type": self.config.block_type,
            "seed": self.config.seed,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "receptive_field": self.receptive_field,
            "stride": self.stride,
            "padding": self.padding,
            "tree_depth": self.tree_depth,
            "n_inputs_per_node": self.n_inputs_per_node,
        }

    def get_output_shape(self, input_shape: torch.Size) -> Optional[torch.Size]:
        """
        Calculate output shape given input shape.

        Args:
            input_shape: Shape of input tensor (batch, channels, height, width)

        Returns:
            Output shape (batch, out_channels, out_height, out_width)
        """
        if len(input_shape) != 4:
            return None

        batch_size, in_channels, in_h, in_w = input_shape

        if in_channels != self.in_channels:
            return None

        # Calculate output spatial dimensions
        out_h = (in_h + 2 * self.padding[0] - self.receptive_field[0]) // self.stride[0] + 1
        out_w = (in_w + 2 * self.padding[1] - self.receptive_field[1]) // self.stride[1] + 1

        return torch.Size([batch_size, self.out_channels, out_h, out_w])

    def _process_patches_through_trees(self, patches):
        """
        Process a batch of patches through all trees.

        Parameters:
        - patches: Tensor of shape (num_patches, patch_size)
        """
        outputs = []
        for tree_layers in self.trees:
            x_tree = patches
            for layer in tree_layers:
                x_tree = layer(x_tree)
            # x_tree is now (num_patches, 1)
            outputs.append(x_tree)

        # Stack outputs: list of (num_patches, 1) -> (num_patches, out_channels)
        return torch.cat(outputs, dim=1)

    def __repr__(self) -> str:
        """Simple model overview."""
        return (
            f"ConvolutionalLayer(\n"
            f"  receptive_field={self.receptive_field}, stride={self.stride}, padding={self.padding}\n"
            f"  in_channels={self.in_channels}, out_channels={self.out_channels}\n"
            f"  tree_architecture={self.hidden_layers}\n"
            f"  node_type={self.node_type.__name__ if hasattr(self.node_type, '__name__') else self.node_type}\n"
            f"  layer_type={self.layer_type.__name__ if hasattr(self.layer_type, '__name__') else self.layer_type}\n"
            f"  n_inputs_per_node={self.n_inputs_per_node}, tree_depth={self.tree_depth}\n"
            f"  num_trees={len(self.trees)}\n"
            f")"
        )

    def forward(self, x):
        """
        Forward pass through convolutional LUT block.

        Parameters:
        - x: Input tensor of shape (batch, in_channels, height, width)
        """
        batch_size = x.shape[0]
        input_h, input_w = x.shape[2], x.shape[3]

        # Calculate output spatial dimensions
        out_h = (input_h + 2 * self.padding[0] - self.receptive_field[0]) // self.stride[0] + 1
        out_w = (input_w + 2 * self.padding[1] - self.receptive_field[1]) // self.stride[1] + 1

        # Extract patches: (batch, patch_size, num_patches)
        # patch_size = in_channels * receptive_field[0] * receptive_field[1]
        patches = self.unfold(x)
        num_patches = patches.shape[2]

        # Reshape to (batch*num_patches, patch_size) for processing
        patches = patches.transpose(1, 2).contiguous()
        patches = patches.view(-1, self.input_size)

        # Apply bit-flip augmentation during training
        if self.training and self.flip_probability > 0.0:
            patches = self._apply_bit_flip(patches)

        # Process patches in chunks to reduce memory usage
        # Instead of processing all batch*patches at once (e.g., 450 samples),
        # we process them in smaller chunks (e.g., 100 samples at a time).
        # This reduces peak memory proportionally to chunk size.

        if self.patch_chunk_size is None or self.patch_chunk_size >= patches.shape[0]:
            # Process all patches at once (original behavior)
            x_out = self._process_patches_through_trees(patches)
        else:
            # Process patches in chunks
            num_patches_total = patches.shape[0]
            chunk_outputs = []

            for chunk_start in range(0, num_patches_total, self.patch_chunk_size):
                chunk_end = min(chunk_start + self.patch_chunk_size, num_patches_total)
                patch_chunk = patches[chunk_start:chunk_end]

                # Process this chunk through all trees
                chunk_out = self._process_patches_through_trees(patch_chunk)
                chunk_outputs.append(chunk_out)

            # Concatenate all chunk outputs
            x_out = torch.cat(chunk_outputs, dim=0)

        # Reshape to (batch, num_patches, out_channels)
        x_out = x_out.view(batch_size, num_patches, self.out_channels)

        # Transpose to (batch, out_channels, num_patches)
        x_out = x_out.transpose(1, 2)

        # Reshape back to spatial format (batch, out_channels, out_h, out_w)
        output = x_out.view(batch_size, self.out_channels, out_h, out_w)

        # Register gradient stabilization hook if enabled
        if self.grad_stabilization != "none" and self.training and output.requires_grad:

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
