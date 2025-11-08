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
    compare_cpu_gpu_forward,
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

    def test_shape_correct(self, node_name):
        """Test 1.1: Forward pass produces correct output shape."""
        node_class = REGISTRY.get_node(node_name)

        with IgnoreWarnings():
            node = instantiate_node(node_class, input_dim=4, output_dim=2)

        # Input shape: (batch_size, input_dim)
        batch_size = 8
        input_tensor = generate_uniform_input((batch_size, 4))

        with torch.no_grad():
            output = node(input_tensor)

        # Output shape should be: (batch_size, output_dim)
        expected_shape = (batch_size, 2)
        assert_shape_equal(output, expected_shape)

    def test_output_range_01(self, node_name):
        """Test 1.2: Output range is [0, 1]."""
        node_class = REGISTRY.get_node(node_name)

        with IgnoreWarnings():
            node = instantiate_node(node_class, input_dim=4, output_dim=2)
        node.eval()

        # Test multiple random inputs
        for seed in [42, 123, 456]:
            input_tensor = generate_uniform_input((8, 4), seed=seed)
            with torch.no_grad():
                output = node(input_tensor)

            assert_range(output, 0.0, 1.0)

    @pytest.mark.gpu
    def test_cpu_gpu_consistency(self, node_name):
        """Test 1.3: CPU and GPU implementations give same forward pass."""
        if not is_cuda_available():
            pytest.skip("CUDA not available")

        node_class = REGISTRY.get_node(node_name)

        with IgnoreWarnings():
            node_cpu = instantiate_node(node_class, input_dim=4, output_dim=2)

        input_cpu = generate_uniform_input((8, 4), seed=42)

        try:
            output_cpu, output_gpu = compare_cpu_gpu_forward(
                node_cpu, input_cpu, atol=CPU_GPU_ATOL, rtol=CPU_GPU_RTOL
            )
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                pytest.skip(f"CUDA error for {node_name}: {e}")
            raise

    def test_gradients_exist(self, node_name):
        """Test 1.4: Gradients exist and are not all zero."""
        node_class = REGISTRY.get_node(node_name)

        with IgnoreWarnings():
            node = instantiate_node(node_class, input_dim=4, output_dim=2)

        node.train()
        input_tensor = generate_uniform_input((8, 4), seed=42)
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
def test_node_different_batch_sizes(node_name):
    """Test node works with different batch sizes."""
    node_class = REGISTRY.get_node(node_name)

    with IgnoreWarnings():
        node = instantiate_node(node_class, input_dim=4, output_dim=2)
    node.eval()

    for batch_size in [1, 8, 32, 128]:
        input_tensor = generate_uniform_input((batch_size, 4), seed=42)
        with torch.no_grad():
            output = node(input_tensor)

        assert output.shape == (batch_size, 2), f"{node_name} failed for batch_size={batch_size}"


@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
def test_node_different_dimensions(node_name):
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
            node = instantiate_node(node_class, input_dim=input_dim, output_dim=output_dim)

        input_tensor = generate_uniform_input((8, input_dim), seed=42)
        with torch.no_grad():
            output = node(input_tensor)

        assert output.shape == (
            8,
            output_dim,
        ), f"{node_name} failed for dims ({input_dim}, {output_dim})"
