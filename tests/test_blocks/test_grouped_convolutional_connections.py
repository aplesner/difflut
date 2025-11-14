"""
Test for grouped convolutional connections.

Verifies that grouped_connections=True ensures full channel coverage
across input channels, while grouped_connections=False may not cover all channels.
Tests across multiple seeds to ensure robustness.
"""

import warnings

# Ignore general runtime warnings in tests
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pytest
import torch
from testing_utils import generate_uniform_input

from difflut.utils.warnings import CUDAWarning, DefaultValueWarning

# Suppress DefaultValueWarnings and CUDAWarnings
warnings.filterwarnings("ignore", category=DefaultValueWarning)
warnings.filterwarnings("ignore", category=CUDAWarning)


@pytest.mark.parametrize("seed", [42, 43, 44, 45, 100])
def test_grouped_connections_coverage(seed):
    """Test that grouped connections ensure full channel coverage across multiple seeds."""
    from difflut import REGISTRY
    from difflut.blocks import ConvolutionalLayer, ConvolutionConfig
    from difflut.layers import LayerConfig
    from difflut.nodes.node_config import NodeConfig

    conv_config = ConvolutionConfig(
        tree_depth=1,
        in_channels=64,
        out_channels=16,
        receptive_field=3,
        stride=1,
        padding=0,
        chunk_size=128,  # Larger than out_channels to disable chunking
        seed=seed,
    )

    layer_cfg = LayerConfig()

    node_type = REGISTRY.get_node("probabilistic")
    layer_type = REGISTRY.get_layer("random")
    conv_layer_type = REGISTRY.get_block("convolutional")

    # Create node config
    node_config = NodeConfig(input_dim=6, output_dim=1)

    # Create convolutional layer with grouped connections
    conv_layer_grouped = conv_layer_type(
        convolution_config=conv_config,
        node_type=node_type,
        node_kwargs=node_config,
        layer_type=layer_type,
        n_inputs_per_node=6,
        layer_config=layer_cfg,
        grouped_connections=True,
        ensure_full_coverage=True,
    )

    # Create convolutional layer without grouped connections
    conv_layer_nongrouped = conv_layer_type(
        convolution_config=conv_config,
        node_type=node_type,
        node_kwargs=node_config,
        layer_type=layer_type,
        n_inputs_per_node=6,
        layer_config=layer_cfg,
        grouped_connections=False,
        ensure_full_coverage=True,
    )

    # Get mapping indices from first layer chunks
    mapping_grouped = conv_layer_grouped.first_layer_chunks[0]._mapping_indices
    mapping_nongrouped = conv_layer_nongrouped.first_layer_chunks[0]._mapping_indices

    # Calculate channel coverage for grouped connections
    # Each index in mapping represents a flattened position in (C, H, W)
    # For receptive_field=3, each spatial position has 3x3=9 elements
    # So channel index = mapping_index // 9
    channel_coverage_grouped = (mapping_grouped[:, 0] // 9).unique().numel()

    # Calculate channel coverage for non-grouped connections
    channel_coverage_nongrouped = (mapping_nongrouped[:, 0] // 9).unique().numel()

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
    from difflut.blocks import ConvolutionalLayer, ConvolutionConfig
    from difflut.layers import LayerConfig
    from difflut.nodes.node_config import NodeConfig

    conv_config = ConvolutionConfig(
        tree_depth=1,
        in_channels=64,
        out_channels=16,
        receptive_field=3,
        stride=1,
        padding=0,
        chunk_size=128,
        seed=seed,
    )

    layer_cfg = LayerConfig()
    node_type = REGISTRY.get_node("probabilistic")
    layer_type = REGISTRY.get_layer("random")
    conv_layer_type = REGISTRY.get_block("convolutional")
    node_config = NodeConfig(input_dim=6, output_dim=1)

    # Create grouped convolutional layer
    conv_layer = conv_layer_type(
        convolution_config=conv_config,
        node_type=node_type,
        node_kwargs=node_config,
        layer_type=layer_type,
        n_inputs_per_node=6,
        layer_config=layer_cfg,
        grouped_connections=True,
        ensure_full_coverage=True,
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
        f"Seed {seed}: Expected output shape {expected_output_shape}, " f"got {output.shape}"
    )

    # Verify output contains valid values (not NaN or Inf)
    assert torch.isfinite(output).all(), f"Seed {seed}: Output contains NaN or Inf values"


@pytest.mark.parametrize("seed", [42, 43])
def test_grouped_connections_gradient_flow(seed):
    """Test that gradients flow through grouped convolutional layers."""
    from difflut import REGISTRY
    from difflut.blocks import ConvolutionalLayer, ConvolutionConfig
    from difflut.layers import LayerConfig
    from difflut.nodes.node_config import NodeConfig

    conv_config = ConvolutionConfig(
        tree_depth=1,
        in_channels=32,
        out_channels=8,
        receptive_field=3,
        stride=1,
        padding=0,
        chunk_size=64,
        seed=seed,
    )

    layer_cfg = LayerConfig()
    node_type = REGISTRY.get_node("probabilistic")
    layer_type = REGISTRY.get_layer("random")
    conv_layer_type = REGISTRY.get_block("convolutional")
    node_config = NodeConfig(input_dim=6, output_dim=1)

    # Create grouped convolutional layer
    conv_layer = conv_layer_type(
        convolution_config=conv_config,
        node_type=node_type,
        node_kwargs=node_config,
        layer_type=layer_type,
        n_inputs_per_node=6,
        layer_config=layer_cfg,
        grouped_connections=True,
        ensure_full_coverage=True,
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
    assert not torch.all(input_tensor.grad == 0), f"Seed {seed}: All input gradients are zero"

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
