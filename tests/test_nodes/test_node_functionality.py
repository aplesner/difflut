"""
Comprehensive forward pass tests for all node types.
Tests: shape correctness, output range [0,1], CPU/GPU consistency, gradients.

Uses pytest parametrization for individual test discovery.
"""

import pytest
import torch
from testing_utils import (
    CPU_GPU_ATOL,
    CPU_GPU_RTOL,
    IgnoreWarnings,
    assert_gradients_exist,
    assert_range,
    assert_shape_equal,
    generate_uniform_input,
    instantiate_node,
)

from difflut.registry import REGISTRY

# ============================================================================
# Node Forward Pass Tests
# ============================================================================


def _get_input_dim_for_node(node_name: str) -> int:
    """Get appropriate input dimension for node type.

    DiffLogic has exponential complexity (2^(2^n)), so we limit it to 2 inputs.
    Other nodes can use larger dimensions.
    """
    if node_name == "difflogic":
        return 2  # 2^(2^2) = 16 functions (manageable)
    return 4  # Default for other nodes


@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
class TestNodeForwardPass:
    """Comprehensive forward pass tests for nodes."""

    def test_shape_correct(self, node_name, device):
        """Test 1.1: Forward pass produces correct output shape."""
        node_class = REGISTRY.get_node(node_name)
        input_dim = _get_input_dim_for_node(node_name)

        with IgnoreWarnings():
            node = instantiate_node(node_class, input_dim=input_dim, output_dim=2).to(device)

        # Input shape: (batch_size, input_dim)
        batch_size = 8
        input_tensor = generate_uniform_input((batch_size, input_dim), device=device)

        with torch.no_grad():
            output = node(input_tensor)

        # Output shape should be: (batch_size, output_dim)
        expected_shape = (batch_size, 2)
        assert_shape_equal(output, expected_shape)

    def test_output_range_01(self, node_name, device):
        """Test 1.2: Output range is [0, 1]."""
        node_class = REGISTRY.get_node(node_name)
        input_dim = _get_input_dim_for_node(node_name)

        with IgnoreWarnings():
            node = instantiate_node(node_class, input_dim=input_dim, output_dim=2).to(device)
        node.eval()

        # Test multiple random inputs
        for seed in [42, 123, 456]:
            input_tensor = generate_uniform_input((8, input_dim), seed=seed, device=device)
            with torch.no_grad():
                output = node(input_tensor)

            assert_range(output, 0.0, 1.0)

    @pytest.mark.gpu
    def test_cpu_gpu_consistency(self, node_name, device):
        """Test 1.3: CPU and GPU implementations give same forward pass results."""
        node_class = REGISTRY.get_node(node_name)
        input_dim = _get_input_dim_for_node(node_name)

        # Set seed before creating nodes to ensure deterministic initialization
        # (important for nodes like FourierNode that use random initialization)
        torch.manual_seed(42)

        with IgnoreWarnings():
            node_cpu = instantiate_node(node_class, input_dim=input_dim, output_dim=2)

        # Reset seed to get same random initialization for GPU node
        torch.manual_seed(42)

        with IgnoreWarnings():
            node_gpu = instantiate_node(node_class, input_dim=input_dim, output_dim=2).cuda()

        # Copy parameters from CPU to GPU (in case there are differences from device transfer)
        node_gpu.load_state_dict(node_cpu.state_dict())

        # Set both to eval mode for deterministic behavior
        node_cpu.eval()
        node_gpu.eval()

        input_cpu = generate_uniform_input((8, input_dim), seed=42, device="cpu")
        input_gpu = input_cpu.cuda()

        with torch.no_grad():
            output_cpu = node_cpu(input_cpu)
            output_gpu = node_gpu(input_gpu).cpu()

        try:
            torch.testing.assert_close(output_cpu, output_gpu, atol=CPU_GPU_ATOL, rtol=CPU_GPU_RTOL)
        except AssertionError as e:
            pytest.fail(f"CPU/GPU outputs differ for {node_name}: {e}")

    def test_gradients_exist(self, node_name, device):
        """Test 1.4: Gradients exist and are not all zero."""
        node_class = REGISTRY.get_node(node_name)
        input_dim = _get_input_dim_for_node(node_name)

        with IgnoreWarnings():
            node = instantiate_node(node_class, input_dim=input_dim, output_dim=2).to(device)

        node.train()
        input_tensor = generate_uniform_input((8, input_dim), seed=42, device=device)
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
def test_node_different_dimensions(node_name, device):
    """
    Test node works with various input/output dimensions and batch sizes.

    This test verifies:
    - Different input dimensions (2, 4, 8, 16)
    - Different output dimensions (1, 2, 4, 8)
    - Various batch sizes (1, 16, 64)
    - Edge cases (single input/output, large dimensions)

    Note: DiffLogic is limited to input_dim=2 due to exponential complexity.
    """
    node_class = REGISTRY.get_node(node_name)

    # DiffLogic has 2^(2^n) complexity - limit to 2 inputs max
    if node_name == "difflogic":
        test_configs = [
            # DiffLogic limited to 2 inputs (16 functions)
            (1, 2, 1),  # Minimal: single sample
            (16, 2, 2),  # Standard: with 2 outputs
            (64, 2, 4),  # Larger batch with more outputs
        ]
    else:
        # Standard test configs for other nodes
        test_configs = [
            # Edge cases
            (1, 2, 1),  # Minimal: single sample, small dims
            (1, 16, 8),  # Single sample, large dims
            # Common configurations
            (16, 4, 1),  # Standard: single output
            (16, 4, 2),  # Standard: dual output
            (16, 8, 4),  # Larger dims
            # Stress tests
            (64, 16, 8),  # Large batch, large dims
            (128, 4, 2),  # Very large batch
        ]

    for batch_size, input_dim, output_dim in test_configs:
        with IgnoreWarnings():
            node = instantiate_node(node_class, input_dim=input_dim, output_dim=output_dim).to(
                device
            )
        node.eval()

        input_tensor = generate_uniform_input((batch_size, input_dim), seed=42, device=device)

        with torch.no_grad():
            output = node(input_tensor)

        # Verify correct output shape
        expected_shape = (batch_size, output_dim)
        assert output.shape == expected_shape, (
            f"{node_name} failed for config "
            f"(batch={batch_size}, in={input_dim}, out={output_dim}): "
            f"expected shape {expected_shape}, got {output.shape}"
        )

        # Verify output is still in valid range
        assert_range(
            output,
            0.0,
            1.0,
            msg=f"{node_name} config (batch={batch_size}, in={input_dim}, out={output_dim})",
        )
