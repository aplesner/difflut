"""
Tests for utility modules like GroupSum.
Verifies GroupSum works correctly for feature grouping and summing.

Uses pytest parametrization for comprehensive testing.
"""

import pytest
import torch
import torch.nn as nn
from testing_utils import (
    assert_gradients_exist,
    assert_shape_equal,
    generate_uniform_input,
)

# ============================================================================
# GroupSum Module Tests
# ============================================================================


class TestGroupSum:
    """Test suite for GroupSum module."""

    def test_basic_forward_pass(self):
        """Test GroupSum basic forward pass."""
        from difflut.utils.modules import GroupSum

        groupsum = GroupSum(k=10, tau=1.0, use_randperm=False)

        # Input: (batch_size, 10) features
        # Output: (batch_size, 10) groups
        input_tensor = generate_uniform_input((4, 10), seed=42)

        output = groupsum(input_tensor)

        assert_shape_equal(output, (4, 10))

    def test_grouping_correctness(self):
        """Test that GroupSum correctly groups and sums features."""
        from difflut.utils.modules import GroupSum

        # Create input where each group has known values
        # k=2 groups, 4 features per group, so 8 features total
        # Group 0: [1, 1, 1, 1] -> sum = 4
        # Group 1: [2, 2, 2, 2] -> sum = 8
        groupsum = GroupSum(k=2, tau=1.0, use_randperm=False)

        input_tensor = torch.tensor([[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]])

        output = groupsum(input_tensor)

        expected = torch.tensor([[4.0, 8.0]])
        assert torch.allclose(output, expected), f"Expected {expected}, got {output}"

    @pytest.mark.parametrize("tau", [0.5, 1.0, 2.0])
    def test_tau_scaling(self, tau):
        """Test that tau parameter correctly scales output."""
        from difflut.utils.modules import GroupSum

        # Create input: [4, 4] -> with different tau values
        input_tensor = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])

        groupsum = GroupSum(k=2, tau=tau, use_randperm=False)
        output = groupsum(input_tensor)

        # output should be [4/tau, 4/tau]
        expected_value = 4.0 / tau
        expected = torch.tensor([[expected_value, expected_value]])
        assert torch.allclose(
            output, expected, atol=1e-5
        ), f"With tau={tau}: expected {expected}, got {output}"

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
    def test_batch_handling(self, batch_size):
        """Test GroupSum with different batch sizes."""
        from difflut.utils.modules import GroupSum

        groupsum = GroupSum(k=5, tau=1.0, use_randperm=False)

        # Create input with correct feature dimension
        # k=5 groups, need multiple of 5 features
        input_tensor = generate_uniform_input((batch_size, 25), seed=42)

        output = groupsum(input_tensor)

        expected_shape = (batch_size, 5)
        assert_shape_equal(output, expected_shape)

    def test_with_layer_output(self):
        """Test GroupSum with realistic layer output."""
        from testing_utils import IgnoreWarnings, instantiate_layer

        from difflut.registry import REGISTRY
        from difflut.utils.modules import GroupSum

        # Create a layer
        layer_class = REGISTRY.get_layer("random")
        with IgnoreWarnings():
            layer = instantiate_layer(
                layer_class, input_size=256, output_size=100, n=4  # 100 nodes
            )

        # Create GroupSum for 10 classes
        groupsum = GroupSum(
            k=10, tau=1.0, use_randperm=False
        )  # 100 nodes / 10 classes = 10 nodes per class

        # Forward pass
        input_tensor = generate_uniform_input((8, 256), seed=42)

        with torch.no_grad():
            layer_output = layer(input_tensor)
            grouped_output = groupsum(layer_output)

        # Output should be (batch_size, k)
        expected_shape = (8, 10)
        assert_shape_equal(grouped_output, expected_shape)

    def test_gradient_flow(self):
        """Test that gradients flow through GroupSum."""
        from difflut.utils.modules import GroupSum

        groupsum = GroupSum(k=5, tau=1.0, use_randperm=False)

        input_tensor = generate_uniform_input((8, 25), seed=42)
        input_tensor.requires_grad = True

        output = groupsum(input_tensor)
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0), "All gradients are zero"

    def test_uneven_grouping(self):
        """Test GroupSum with non-divisible feature counts (should pad or handle)."""
        from difflut.utils.modules import GroupSum

        # k=3 groups, 10 features -> not evenly divisible
        groupsum = GroupSum(k=3, tau=1.0, use_randperm=False)

        input_tensor = generate_uniform_input((4, 10), seed=42)

        # Should still work (either by padding or other strategy)
        output = groupsum(input_tensor)

        # Output should still be (batch_size, k)
        assert output.shape[0] == 4
        assert output.shape[1] == 3


# ============================================================================
# GroupSum Edge Cases
# ============================================================================


class TestGroupSumEdgeCases:
    """Test edge cases for GroupSum module."""

    def test_single_group(self):
        """Test GroupSum with k=1 (single group)."""
        from difflut.utils.modules import GroupSum

        groupsum = GroupSum(k=1, tau=1.0, use_randperm=False)
        input_tensor = generate_uniform_input((4, 10), seed=42)

        output = groupsum(input_tensor)

        # Should sum all features
        expected = input_tensor.sum(dim=1, keepdim=True)
        assert torch.allclose(output, expected, atol=1e-5)

    def test_many_groups(self):
        """Test GroupSum with many groups."""
        from difflut.utils.modules import GroupSum

        groupsum = GroupSum(k=100, tau=1.0, use_randperm=False)
        input_tensor = generate_uniform_input((4, 100), seed=42)

        output = groupsum(input_tensor)

        assert_shape_equal(output, (4, 100))

    @pytest.mark.parametrize("use_randperm", [True, False])
    def test_randperm_parameter(self, use_randperm):
        """Test GroupSum with use_randperm option."""
        from difflut.utils.modules import GroupSum

        groupsum = GroupSum(k=5, tau=1.0, use_randperm=use_randperm)
        input_tensor = generate_uniform_input((4, 25), seed=42)

        output = groupsum(input_tensor)

        # Should work regardless of use_randperm
        assert_shape_equal(output, (4, 5))

    def test_zero_input(self):
        """Test GroupSum with zero input."""
        from difflut.utils.modules import GroupSum

        groupsum = GroupSum(k=5, tau=1.0, use_randperm=False)
        input_tensor = torch.zeros(4, 25)

        output = groupsum(input_tensor)

        # Output should be all zeros
        assert torch.allclose(output, torch.zeros(4, 5))

    def test_one_input(self):
        """Test GroupSum with all-ones input."""
        from difflut.utils.modules import GroupSum

        groupsum = GroupSum(k=5, tau=1.0, use_randperm=False)
        input_tensor = torch.ones(4, 25)

        output = groupsum(input_tensor)

        # Each group should sum to 5 (25 features / 5 groups = 5 features per group)
        expected = torch.ones(4, 5) * 5.0
        assert torch.allclose(output, expected, atol=1e-5)
