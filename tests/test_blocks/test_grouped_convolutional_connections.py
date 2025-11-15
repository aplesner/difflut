"""
Test for grouped convolutional connections.

Verifies that grouped_connections=True ensures full channel coverage
across input channels, while grouped_connections=False may not cover all channels.
Tests across multiple seeds to ensure robustness.
"""

import warnings

import pytest
import torch
from testing_utils import generate_uniform_input

from difflut.utils.warnings import CUDAWarning, DefaultValueWarning

# Ignore general runtime warnings in tests
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Suppress DefaultValueWarnings and CUDAWarnings
warnings.filterwarnings("ignore", category=DefaultValueWarning)
warnings.filterwarnings("ignore", category=CUDAWarning)


@pytest.mark.parametrize("seed", [42, 43, 44, 45, 100])
def test_grouped_connections_coverage(seed, device):
    """Test that grouped connections ensure full channel coverage across multiple seeds."""
    from difflut import REGISTRY
    from difflut.blocks import BlockConfig, ConvolutionalLayer
    from difflut.layers.layer_config import LayerConfig
    from difflut.nodes.node_config import NodeConfig

    block_config = BlockConfig(
        block_type="convolutional",
        seed=seed,
        tree_depth=1,
        in_channels=64,
        out_channels=16,
        receptive_field=(3, 3),
        stride=(1, 1),
        padding=(0, 0),
        n_inputs_per_node=6,
        node_kwargs={"input_dim": 6, "output_dim": 1},
        grouped_connections=True,
        ensure_full_coverage=True,
    )

    node_type = REGISTRY.get_node("probabilistic")
    layer_type = REGISTRY.get_layer("random")
    conv_layer_type = REGISTRY.get_block("convolutional")

    # Create convolutional layer with grouped connections
    conv_layer_grouped = conv_layer_type(
        config=block_config,
        node_type=node_type,
        layer_type=layer_type,
    )

    # Create block config without grouped connections for comparison
    block_config_nongrouped = BlockConfig(
        block_type="convolutional",
        seed=seed,
        tree_depth=1,
        in_channels=64,
        out_channels=16,
        receptive_field=(3, 3),
        stride=(1, 1),
        padding=(0, 0),
        n_inputs_per_node=6,
        node_kwargs={"input_dim": 6, "output_dim": 1},
        grouped_connections=False,
        ensure_full_coverage=True,
    )

    # Create convolutional layer without grouped connections
    conv_layer_nongrouped = conv_layer_type(
        config=block_config_nongrouped,
        node_type=node_type,
        layer_type=layer_type,
    )

    # Get mapping indices from first layer of all trees (not just first tree)
    # In the new architecture, trees[i] is a ModuleList of layers for tree i
    # trees[i][0] is the first layer of tree i

    # Collect all mapping indices from all trees to see channel coverage
    all_indices_grouped = []
    all_indices_nongrouped = []

    for tree_idx in range(len(conv_layer_grouped.trees)):
        mapping_grouped_tree = conv_layer_grouped.trees[tree_idx][0]._mapping_indices
        all_indices_grouped.append(mapping_grouped_tree)

        mapping_nongrouped_tree = conv_layer_nongrouped.trees[tree_idx][
            0
        ]._mapping_indices
        all_indices_nongrouped.append(mapping_nongrouped_tree)

    # Concatenate all indices from all trees
    all_indices_grouped = torch.cat(all_indices_grouped, dim=0)
    all_indices_nongrouped = torch.cat(all_indices_nongrouped, dim=0)

    # Calculate channel coverage for grouped connections
    # Each index in mapping represents a flattened position in the patch
    # For receptive_field=3 with 64 input channels:
    # Patch flattened size = 64 * 9 (3x3 spatial)
    # Channel index = mapping_index // 9
    channel_coverage_grouped = (all_indices_grouped.flatten() // 9).unique().numel()

    # Calculate channel coverage for non-grouped connections
    channel_coverage_nongrouped = (
        (all_indices_nongrouped.flatten() // 9).unique().numel()
    )

    # Grouped connections should cover all 64 channels
    assert channel_coverage_grouped == 64, (
        f"Seed {seed}: Grouped connections should cover all 64 channels, "
        f"but only covered {channel_coverage_grouped}"
    )

    # Non-grouped connections typically cover fewer channels (not guaranteed to cover all)
    # This is expected behavior - we just verify it's a valid number
    assert 1 <= channel_coverage_nongrouped <= 64, (
        f"Seed {seed}: Non-grouped channel coverage {channel_coverage_nongrouped} "
        f"is outside valid range [1, 64]"
    )


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_grouped_connections_forward_pass(seed):
    """Test that grouped convolutional layers can perform forward pass."""
    from difflut import REGISTRY
    from difflut.blocks import BlockConfig, ConvolutionalLayer
    from difflut.layers.layer_config import LayerConfig
    from difflut.nodes.node_config import NodeConfig

    block_config = BlockConfig(
        block_type="convolutional",
        seed=seed,
        tree_depth=1,
        in_channels=64,
        out_channels=16,
        receptive_field=(3, 3),
        stride=(1, 1),
        padding=(0, 0),
        n_inputs_per_node=6,
        node_kwargs={"input_dim": 6, "output_dim": 1},
        grouped_connections=True,
        ensure_full_coverage=True,
    )

    node_type = REGISTRY.get_node("probabilistic")
    layer_type = REGISTRY.get_layer("random")
    conv_layer_type = REGISTRY.get_block("convolutional")

    # Create grouped convolutional layer
    conv_layer = conv_layer_type(
        config=block_config,
        node_type=node_type,
        layer_type=layer_type,
    )

    # Create input tensor: (batch_size, in_channels, height, width)
    # With receptive_field=3, padding=0, stride=1:
    # output size = (input_size - receptive_field) / stride + 1
    # For output 14x14, we need input 16x16
    input_tensor = generate_uniform_input((4, 64, 16, 16), seed=seed)

    # Forward pass
    output = conv_layer(input_tensor)

    # Verify output shape: (batch_size, out_channels, out_height, out_width)
    expected_output_shape = (4, 16, 14, 14)
    assert output.shape == expected_output_shape, (
        f"Seed {seed}: Expected output shape {expected_output_shape}, "
        f"got {output.shape}"
    )

    # Verify output contains valid values (not NaN or Inf)
    assert torch.isfinite(
        output
    ).all(), f"Seed {seed}: Output contains NaN or Inf values"


@pytest.mark.parametrize("seed", [42, 43])
def test_grouped_connections_gradient_flow(seed):
    """Test that gradients flow through grouped convolutional layers."""
    from difflut import REGISTRY
    from difflut.blocks import BlockConfig, ConvolutionalLayer
    from difflut.layers.layer_config import LayerConfig
    from difflut.nodes.node_config import NodeConfig

    block_config = BlockConfig(
        block_type="convolutional",
        seed=seed,
        tree_depth=1,
        in_channels=32,
        out_channels=8,
        receptive_field=(3, 3),
        stride=(1, 1),
        padding=(0, 0),
        n_inputs_per_node=6,
        node_kwargs={"input_dim": 6, "output_dim": 1},
        grouped_connections=True,
        ensure_full_coverage=True,
    )

    node_type = REGISTRY.get_node("probabilistic")
    layer_type = REGISTRY.get_layer("random")
    conv_layer_type = REGISTRY.get_block("convolutional")

    # Create grouped convolutional layer
    conv_layer = conv_layer_type(
        config=block_config,
        node_type=node_type,
        layer_type=layer_type,
    )

    # Create input tensor with requires_grad=True
    input_tensor = generate_uniform_input((2, 32, 8, 8), seed=seed)
    input_tensor.requires_grad = True

    # Forward pass
    output = conv_layer(input_tensor)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check that input has gradients
    assert (
        input_tensor.grad is not None
    ), f"Seed {seed}: Input tensor should have gradients after backward pass"
    assert not torch.all(
        input_tensor.grad == 0
    ), f"Seed {seed}: All input gradients are zero"

    # Check that layer parameters have gradients
    param_count = 0
    params_with_grads = 0
    for param in conv_layer.parameters():
        param_count += 1
        if param.grad is not None and param.grad.abs().sum() > 0:
            params_with_grads += 1

    assert params_with_grads > 0, (
        f"Seed {seed}: No layer parameters have non-zero gradients "
        f"({params_with_grads}/{param_count} parameters have gradients)"
    )
