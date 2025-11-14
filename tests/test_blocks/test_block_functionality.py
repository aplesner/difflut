"""
Comprehensive forward pass tests for all block types.
Tests: shape correctness, output range [0,1], CPU/GPU consistency, gradients.

Uses pytest parametrization for individual test discovery.
"""

import pytest
import torch
import torch.nn as nn
from testing_utils import (
    CPU_GPU_ATOL,
    CPU_GPU_RTOL,
    IgnoreWarnings,
    assert_gradients_exist,
    assert_range,
    assert_shape_equal,
    generate_uniform_input,
    is_cuda_available,
)

from difflut.registry import REGISTRY
from difflut.blocks import ConvolutionalLayer, ConvolutionConfig
from difflut.layers import LayerConfig
from difflut.nodes.node_config import NodeConfig

# ============================================================================
# Block Forward Pass Tests
# ============================================================================

_testable_blocks = REGISTRY.list_blocks()


def instantiate_block(block_class, node_type, layer_type, seed=42):
    """Instantiate a block with default configuration."""
    conv_config = ConvolutionConfig(
        tree_depth=1,
        in_channels=16,
        out_channels=8,
        receptive_field=3,
        stride=1,
        padding=0,
        seed=seed,
    )

    layer_cfg = LayerConfig()
    node_config = NodeConfig(input_dim=6, output_dim=1)

    return block_class(
        convolution_config=conv_config,
        node_type=node_type,
        node_kwargs=node_config,
        layer_type=layer_type,
        n_inputs_per_node=6,
        layer_config=layer_cfg,
    )


@pytest.mark.parametrize("block_name", _testable_blocks)
class TestBlockForwardPass:
    """Comprehensive forward pass tests for blocks."""

    def test_shape_correct(self, block_name, device):
        """Test 3.1: Forward pass produces correct output shape."""
        block_class = REGISTRY.get_block(block_name)
        node_type = REGISTRY.get_node("probabilistic")
        layer_type = REGISTRY.get_layer("random")

        with IgnoreWarnings():
            block = instantiate_block(block_class, node_type, layer_type).to(device)

        # Input shape: (batch_size, in_channels, height, width)
        # With receptive_field=3, padding=0, stride=1:
        # output size = (input_size - receptive_field) / stride + 1 = (16 - 3) / 1 + 1 = 14
        batch_size = 4
        input_tensor = generate_uniform_input((batch_size, 16, 16, 16), device=device)

        with torch.no_grad():
            output = block(input_tensor)

        # Output shape should be: (batch_size, out_channels, height, width)
        expected_shape = (batch_size, 8, 14, 14)
        assert_shape_equal(output, expected_shape)

    def test_output_range_01(self, block_name, device):
        """Test 3.2: Output range is [0, 1]."""
        block_class = REGISTRY.get_block(block_name)
        node_type = REGISTRY.get_node("probabilistic")
        layer_type = REGISTRY.get_layer("random")

        with IgnoreWarnings():
            block = instantiate_block(block_class, node_type, layer_type).to(device)
        block.eval()

        # Test multiple random inputs
        for seed in [42, 123, 456]:
            input_tensor = generate_uniform_input((4, 16, 16, 16), seed=seed, device=device)
            with torch.no_grad():
                output = block(input_tensor)

            assert_range(output, 0.0, 1.0)

    @pytest.mark.gpu
    def test_cpu_gpu_consistency(self, block_name):
        """Test 3.3: CPU and GPU implementations give same forward pass results."""
        if not is_cuda_available():
            pytest.skip("CUDA not available")

        block_class = REGISTRY.get_block(block_name)
        node_type = REGISTRY.get_node("probabilistic")
        layer_type = REGISTRY.get_layer("random")

        # Set seed before creating blocks to ensure deterministic initialization
        torch.manual_seed(42)

        with IgnoreWarnings():
            block_cpu = instantiate_block(block_class, node_type, layer_type)

        # Reset seed to get same random initialization for GPU block
        torch.manual_seed(42)

        with IgnoreWarnings():
            block_gpu = instantiate_block(block_class, node_type, layer_type).cuda()

        # Copy parameters from CPU to GPU
        block_gpu.load_state_dict(block_cpu.state_dict())

        input_cpu = generate_uniform_input((2, 16, 16, 16), seed=42, device="cpu")
        input_gpu = input_cpu.cuda()

        with torch.no_grad():
            output_cpu = block_cpu(input_cpu)
            output_gpu = block_gpu(input_gpu).cpu()

        try:
            torch.testing.assert_close(output_cpu, output_gpu, atol=CPU_GPU_ATOL, rtol=CPU_GPU_RTOL)
        except AssertionError as e:
            pytest.fail(f"CPU/GPU outputs differ for {block_name}: {e}")

    def test_gradients_exist(self, block_name, device):
        """Test 3.4: Gradients exist and are not all zero."""
        block_class = REGISTRY.get_block(block_name)
        node_type = REGISTRY.get_node("probabilistic")
        layer_type = REGISTRY.get_layer("random")

        with IgnoreWarnings():
            block = instantiate_block(block_class, node_type, layer_type).to(device)

        block.train()
        input_tensor = generate_uniform_input((2, 16, 16, 16), seed=42, device=device)
        input_tensor.requires_grad = True

        output = block(input_tensor)
        loss = output.sum()
        loss.backward()

        # Check gradients exist for parameters
        assert_gradients_exist(block)


# ============================================================================
# Block with Different Node Types
# ============================================================================


@pytest.mark.parametrize("block_name", _testable_blocks)
@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
def test_block_with_node_type(block_name, node_name, device):
    """Test block works with all node types."""
    block_class = REGISTRY.get_block(block_name)
    node_class = REGISTRY.get_node(node_name)
    layer_type = REGISTRY.get_layer("random")

    try:
        with IgnoreWarnings():
            block = instantiate_block(block_class, node_class, layer_type).to(device)

        # Test forward pass
        input_tensor = generate_uniform_input((2, 16, 16, 16), seed=42, device=device)
        with torch.no_grad():
            output = block(input_tensor)

        assert output.shape == (2, 8, 14, 14), (
            f"{block_name} with {node_name} produced wrong shape {output.shape}"
        )

    except TypeError as e:
        # Some block/node combinations might not be compatible
        if "node_type" in str(e):
            pytest.skip(f"{block_name} does not support node_type parameter")
        raise


# ============================================================================
# Additional Block Tests
# ============================================================================


@pytest.mark.parametrize("block_name", _testable_blocks)
def test_block_different_sizes(block_name, device):
    """Test block works with different input/output channel sizes."""
    block_class = REGISTRY.get_block(block_name)
    node_type = REGISTRY.get_node("probabilistic")
    layer_type = REGISTRY.get_layer("random")

    test_configs = [
        (8, 4),  # Small
        (16, 8),  # Medium
        (32, 16),  # Large
    ]

    for in_channels, out_channels in test_configs:
        conv_config = ConvolutionConfig(
            tree_depth=1,
            in_channels=in_channels,
            out_channels=out_channels,
            receptive_field=3,
            stride=1,
            padding=0,
            seed=42,
        )

        layer_cfg = LayerConfig()
        node_config = NodeConfig(input_dim=6, output_dim=1)

        with IgnoreWarnings():
            block = block_class(
                convolution_config=conv_config,
                node_type=node_type,
                node_kwargs=node_config,
                layer_type=layer_type,
                n_inputs_per_node=6,
                layer_config=layer_cfg,
            ).to(device)

        input_tensor = generate_uniform_input((2, in_channels, 16, 16), seed=42, device=device)
        with torch.no_grad():
            output = block(input_tensor)

        assert output.shape == (2, out_channels, 14, 14), (
            f"{block_name} failed for sizes ({in_channels} → {out_channels})"
        )


@pytest.mark.parametrize("block_name", _testable_blocks)
def test_block_grouped_connections(block_name, device):
    """Test block with grouped connections option."""
    block_class = REGISTRY.get_block(block_name)
    node_type = REGISTRY.get_node("probabilistic")
    layer_type = REGISTRY.get_layer("random")

    conv_config = ConvolutionConfig(
        tree_depth=1,
        in_channels=32,
        out_channels=8,
        receptive_field=3,
        stride=1,
        padding=0,
        seed=42,
    )

    layer_cfg = LayerConfig()
    node_config = NodeConfig(input_dim=6, output_dim=1)

    with IgnoreWarnings():
        block = block_class(
            convolution_config=conv_config,
            node_type=node_type,
            node_kwargs=node_config,
            layer_type=layer_type,
            n_inputs_per_node=6,
            layer_config=layer_cfg,
            grouped_connections=True,
            ensure_full_coverage=True,
        ).to(device)

    input_tensor = generate_uniform_input((2, 32, 16, 16), seed=42, device=device)
    with torch.no_grad():
        output = block(input_tensor)

    assert output.shape == (2, 8, 14, 14), (
        f"{block_name} with grouped_connections failed, got shape {output.shape}"
    )
