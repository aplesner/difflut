"""
Comprehensive forward pass tests for all node types.
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
    instantiate_node,
    is_cuda_available,
)

from difflut.registry import REGISTRY

# ============================================================================
# Node Forward Pass Tests
# ============================================================================


@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
class TestNodeForwardPass:
    """Comprehensive forward pass tests for nodes."""

    def test_shape_correct(self, node_name, device):
        """Test 1.1: Forward pass produces correct output shape."""
        node_class = REGISTRY.get_node(node_name)

        with IgnoreWarnings():
            node = instantiate_node(node_class, input_dim=4, output_dim=2).to(device)

        # Input shape: (batch_size, input_dim)
        batch_size = 8
        input_tensor = generate_uniform_input((batch_size, 4), device=device)

        with torch.no_grad():
            output = node(input_tensor)

        # Output shape should be: (batch_size, output_dim)
        expected_shape = (batch_size, 2)
        assert_shape_equal(output, expected_shape)

    def test_output_range_01(self, node_name, device):
        """Test 1.2: Output range is [0, 1]."""
        node_class = REGISTRY.get_node(node_name)

        with IgnoreWarnings():
            node = instantiate_node(node_class, input_dim=4, output_dim=2).to(device)
        node.eval()

        # Test multiple random inputs
        for seed in [42, 123, 456]:
            input_tensor = generate_uniform_input((8, 4), seed=seed, device=device)
            with torch.no_grad():
                output = node(input_tensor)

            assert_range(output, 0.0, 1.0)

    @pytest.mark.gpu
    def test_cpu_gpu_consistency(self, node_name):
        """Test 1.3: CPU and GPU implementations give same forward pass results."""
        if not is_cuda_available():
            pytest.fail("Test marked with @pytest.mark.gpu but CUDA is not available")

        node_class = REGISTRY.get_node(node_name)

        with IgnoreWarnings():
            node_cpu = instantiate_node(node_class, input_dim=4, output_dim=2)
            node_gpu = instantiate_node(node_class, input_dim=4, output_dim=2).cuda()

        # Copy parameters from CPU to GPU
        node_gpu.load_state_dict(node_cpu.state_dict())
        
        # Set both to eval mode for deterministic behavior
        node_cpu.eval()
        node_gpu.eval()

        input_cpu = generate_uniform_input((8, 4), seed=42, device="cpu")
        input_gpu = input_cpu.cuda()

        with torch.no_grad():
            output_cpu = node_cpu(input_cpu)
            output_gpu = node_gpu(input_gpu).cpu()

        try:
            torch.testing.assert_close(
                output_cpu, output_gpu, atol=CPU_GPU_ATOL, rtol=CPU_GPU_RTOL
            )
        except AssertionError as e:
            pytest.fail(f"CPU/GPU outputs differ for {node_name}: {e}")

    def test_gradients_exist(self, node_name, device):
        """Test 1.4: Gradients exist and are not all zero."""
        node_class = REGISTRY.get_node(node_name)

        with IgnoreWarnings():
            node = instantiate_node(node_class, input_dim=4, output_dim=2).to(device)

        node.train()
        input_tensor = generate_uniform_input((8, 4), seed=42, device=device)
        input_tensor.requires_grad = True

        output = node(input_tensor)
        loss = output.sum()
        loss.backward()

        # Check gradients exist for parameters
        assert_gradients_exist(node)


# ============================================================================
# Additional Node Tests
# ============================================================================


@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
def test_node_different_batch_sizes(node_name, device):
    """Test node works with different batch sizes."""
    node_class = REGISTRY.get_node(node_name)

    with IgnoreWarnings():
        node = instantiate_node(node_class, input_dim=4, output_dim=2).to(device)
    node.eval()

    for batch_size in [1, 8, 32, 128]:
        input_tensor = generate_uniform_input((batch_size, 4), seed=42, device=device)
        with torch.no_grad():
            output = node(input_tensor)

        assert output.shape == (batch_size, 2), f"{node_name} failed for batch_size={batch_size}"


@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
def test_node_different_dimensions(node_name, device):
    """Test node works with different input/output dimensions."""
    node_class = REGISTRY.get_node(node_name)

    test_configs = [
        (2, 1),  # Small
        (4, 2),  # Medium
        (8, 4),  # Large
        (16, 8),  # Very large
    ]

    for input_dim, output_dim in test_configs:
        with IgnoreWarnings():
            node = instantiate_node(node_class, input_dim=input_dim, output_dim=output_dim).to(
                device
            )

        input_tensor = generate_uniform_input((8, input_dim), seed=42, device=device)
        with torch.no_grad():
            output = node(input_tensor)

        assert output.shape == (
            8,
            output_dim,
        ), f"{node_name} failed for dims ({input_dim}, {output_dim})"
