"""
Comprehensive forward pass tests for all layer types.
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
    instantiate_layer,
)

from difflut.registry import REGISTRY

# ============================================================================
# Layer Forward Pass Tests
# ============================================================================

_testable_layers = REGISTRY.list_layers()


@pytest.mark.parametrize("layer_name", _testable_layers)
class TestLayerForwardPass:
    """Comprehensive forward pass tests for layers."""

    def test_shape_correct(self, layer_name, device):
        """Test 2.1: Forward pass produces correct output shape."""
        layer_class = REGISTRY.get_layer(layer_name)

        with IgnoreWarnings():
            layer = instantiate_layer(layer_class, input_size=256, output_size=128, n=4).to(device)

        # Input shape: (batch_size, input_size)
        batch_size = 8
        input_tensor = generate_uniform_input((batch_size, 256), device=device)

        with torch.no_grad():
            output = layer(input_tensor)

        # Output shape should be: (batch_size, output_size)
        expected_shape = (batch_size, 128)
        assert_shape_equal(output, expected_shape)

    def test_output_range_01(self, layer_name, device):
        """Test 2.2: Output range is [0, 1]."""
        layer_class = REGISTRY.get_layer(layer_name)

        with IgnoreWarnings():
            layer = instantiate_layer(layer_class, input_size=256, output_size=128, n=4).to(device)
        layer.eval()

        # Test multiple random inputs
        for seed in [42, 123, 456]:
            input_tensor = generate_uniform_input((8, 256), seed=seed, device=device)
            with torch.no_grad():
                output = layer(input_tensor)

            assert_range(output, 0.0, 1.0)

    @pytest.mark.gpu
    def test_cpu_gpu_consistency(self, layer_name):
        """Test 2.3: CPU and GPU implementations give same forward pass results."""
        layer_class = REGISTRY.get_layer(layer_name)

        # Set seed before creating layers to ensure deterministic initialization
        # (important for RandomLayer and layers using nodes with random init like FourierNode)
        torch.manual_seed(42)

        with IgnoreWarnings():
            layer_cpu = instantiate_layer(layer_class, input_size=256, output_size=128, n=4)

        # Reset seed to get same random initialization for GPU layer
        torch.manual_seed(42)

        with IgnoreWarnings():
            layer_gpu = instantiate_layer(layer_class, input_size=256, output_size=128, n=4).cuda()

        # Copy parameters from CPU to GPU (in case there are differences from device transfer)
        layer_gpu.load_state_dict(layer_cpu.state_dict())

        input_cpu = generate_uniform_input((8, 256), seed=42, device="cpu")
        input_gpu = input_cpu.cuda()

        with torch.no_grad():
            output_cpu = layer_cpu(input_cpu)
            output_gpu = layer_gpu(input_gpu).cpu()

        try:
            torch.testing.assert_close(output_cpu, output_gpu, atol=CPU_GPU_ATOL, rtol=CPU_GPU_RTOL)
        except AssertionError as e:
            pytest.fail(f"CPU/GPU outputs differ for {layer_name}: {e}")

    def test_gradients_exist(self, layer_name, device):
        """Test 2.4: Gradients exist and are not all zero."""
        layer_class = REGISTRY.get_layer(layer_name)

        with IgnoreWarnings():
            layer = instantiate_layer(layer_class, input_size=256, output_size=128, n=4).to(device)

        layer.train()
        input_tensor = generate_uniform_input((8, 256), seed=42, device=device)
        input_tensor.requires_grad = True

        output = layer(input_tensor)
        loss = output.sum()
        loss.backward()

        # Check gradients exist for parameters
        assert_gradients_exist(layer)


# ============================================================================
# Layer with Different Node Types
# ============================================================================


@pytest.mark.parametrize("layer_name", _testable_layers)
@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
def test_layer_with_node_type(layer_name, node_name, device):
    """Test layer works with all node types."""
    layer_class = REGISTRY.get_layer(layer_name)
    node_class = REGISTRY.get_node(node_name)

    # Use n=2 for difflogic to avoid excessive memory usage
    # (difflogic with n=4 would require 65,536 Boolean functions)
    n = 2 if node_name == "difflogic" else 4

    try:
        with IgnoreWarnings():
            layer = instantiate_layer(
                layer_class, input_size=256, output_size=128, node_type=node_class, n=n
            ).to(device)

        # Test forward pass
        input_tensor = generate_uniform_input((8, 256), seed=42, device=device)
        with torch.no_grad():
            output = layer(input_tensor)

        assert output.shape == (
            8,
            128,
        ), f"{layer_name} with {node_name} produced wrong shape"

    except TypeError as e:
        # Some layer/node combinations might not be compatible
        if "node_type" in str(e):
            pytest.skip(f"{layer_name} does not support node_type parameter")
        raise


# ============================================================================
# Additional Layer Tests
# ============================================================================


@pytest.mark.parametrize("layer_name", _testable_layers)
def test_layer_different_sizes(layer_name, device):
    """Test layer works with different input/output sizes."""
    layer_class = REGISTRY.get_layer(layer_name)

    test_configs = [
        (64, 32),  # Small
        (256, 128),  # Medium
        (512, 256),  # Large
    ]

    for input_size, output_size in test_configs:
        with IgnoreWarnings():
            layer = instantiate_layer(
                layer_class, input_size=input_size, output_size=output_size, n=4
            ).to(device)

        input_tensor = generate_uniform_input((8, input_size), seed=42, device=device)
        with torch.no_grad():
            output = layer(input_tensor)

        assert output.shape == (
            8,
            output_size,
        ), f"{layer_name} failed for sizes ({input_size}, {output_size})"
