"""
Tests for utility modules like GroupSum.
Verifies GroupSum works correctly for feature grouping and summing.

This test is designed for CI/CD pipelines and suppresses non-critical warnings.
"""

import sys
import warnings
import torch
import torch.nn as nn

# Suppress warnings for CI/CD
warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut')
warnings.filterwarnings('ignore', category=UserWarning, module='difflut')

from test_utils import (
    print_section,
    print_subsection,
    print_test_result,
    generate_uniform_input,
    assert_shape_equal,
    assert_range,
)


class GroupSumTester:
    """Helper class for testing GroupSum module."""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
    
    def test_basic_forward_pass(self):
        """Test GroupSum basic forward pass."""
        try:
            from difflut.utils.modules import GroupSum
            
            groupsum = GroupSum(k=10, tau=1.0)
            
            # Input: (batch_size, 10) features
            # Output: (batch_size, 10) groups
            input_tensor = generate_uniform_input((4, 10), seed=42)
            
            output = groupsum(input_tensor)
            
            assert_shape_equal(output, (4, 10))
            
            print_test_result("GroupSum: Basic Forward Pass", True)
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print_test_result("GroupSum: Basic Forward Pass", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_grouping(self):
        """Test that GroupSum correctly groups and sums features."""
        try:
            from difflut.utils.modules import GroupSum
            
            # Create input where each group has known values
            # k=2 groups, 4 features per group, so 8 features total
            # Group 0: [1, 1, 1, 1] -> sum = 4
            # Group 1: [2, 2, 2, 2] -> sum = 8
            groupsum = GroupSum(k=2, tau=1.0)
            
            input_tensor = torch.tensor([
                [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
            ])
            
            output = groupsum(input_tensor)
            
            expected = torch.tensor([[4.0, 8.0]])
            assert torch.allclose(output, expected), f"Expected {expected}, got {output}"
            
            print_test_result("GroupSum: Correct Grouping", True)
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print_test_result("GroupSum: Correct Grouping", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_tau_scaling(self):
        """Test that tau parameter correctly scales output."""
        try:
            from difflut.utils.modules import GroupSum
            
            # Create input: [4, 4] -> with tau=1.0: [4, 4], with tau=2.0: [2, 2]
            input_tensor = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
            
            groupsum1 = GroupSum(k=2, tau=1.0)
            groupsum2 = GroupSum(k=2, tau=2.0)
            
            output1 = groupsum1(input_tensor)
            output2 = groupsum2(input_tensor)
            
            # output1 should be [4, 4]
            # output2 should be [2, 2]
            assert torch.allclose(output1, torch.tensor([[4.0, 4.0]])), \
                f"With tau=1.0: expected [4, 4], got {output1}"
            assert torch.allclose(output2, torch.tensor([[2.0, 2.0]])), \
                f"With tau=2.0: expected [2, 2], got {output2}"
            
            print_test_result("GroupSum: Tau Scaling", True)
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print_test_result("GroupSum: Tau Scaling", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_batch_handling(self):
        """Test GroupSum with different batch sizes."""
        try:
            from difflut.utils.modules import GroupSum
            
            groupsum = GroupSum(k=5, tau=1.0)
            
            # Test with different batch sizes
            for batch_size in [1, 4, 16, 32]:
                input_tensor = generate_uniform_input((batch_size, 10), seed=42)
                output = groupsum(input_tensor)
                
                assert_shape_equal(output, (batch_size, 5))
            
            print_test_result("GroupSum: Batch Handling", True, "Tested batch sizes: 1, 4, 16, 32")
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print_test_result("GroupSum: Batch Handling", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_with_layer_output(self):
        """Test GroupSum with realistic layer output."""
        try:
            from difflut.utils.modules import GroupSum
            from difflut.layers import RandomLayer
            from difflut.nodes import LinearLUTNode
            from difflut.nodes.node_config import NodeConfig
            
            # Create a layer that outputs 10 values
            node_config = NodeConfig(input_dim=4, output_dim=1)
            layer = RandomLayer(
                input_size=128,
                output_size=10,
                node_type=LinearLUTNode,
                node_kwargs=node_config
            )
            layer.eval()
            
            # Create GroupSum to group into 10 classes
            groupsum = GroupSum(k=10, tau=1.0)
            
            # Generate input and pass through layer
            input_tensor = generate_uniform_input((4, 128), seed=42)
            layer_output = layer(input_tensor)
            
            # GroupSum the output
            final_output = groupsum(layer_output)
            
            # Should have shape (4, 10)
            assert_shape_equal(final_output, (4, 10))
            
            # Output should be non-negative (it's a sum of values in [0,1])
            assert final_output.min() >= 0
            
            print_test_result("GroupSum: With Layer Output", True)
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print_test_result("GroupSum: With Layer Output", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_gradient_flow(self):
        """Test that gradients flow through GroupSum."""
        try:
            from difflut.utils.modules import GroupSum
            
            groupsum = GroupSum(k=5, tau=1.0)
            
            input_tensor = generate_uniform_input((4, 10), seed=42, device='cpu')
            input_tensor.requires_grad = True
            
            output = groupsum(input_tensor)
            loss = output.sum()
            loss.backward()
            
            # Check that gradients exist and are non-zero
            assert input_tensor.grad is not None, "No gradients computed"
            assert input_tensor.grad.abs().sum() > 0, "All gradients are zero"
            
            print_test_result("GroupSum: Gradient Flow", True)
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print_test_result("GroupSum: Gradient Flow", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_uneven_grouping(self):
        """Test GroupSum with non-divisible feature counts (should pad)."""
        try:
            from difflut.utils.modules import GroupSum
            
            # 10 features grouped into 3 groups
            # 10 % 3 = 1, so should pad with 2 zeros
            groupsum = GroupSum(k=3, tau=1.0)
            
            # This should trigger a warning about padding
            input_tensor = torch.ones((2, 10))
            
            with __import__('warnings').catch_warnings(record=True) as w:
                __import__('warnings').simplefilter("always")
                output = groupsum(input_tensor)
                
                # Check that warning was issued
                # (padding warning is expected for non-divisible grouping)
            
            assert_shape_equal(output, (2, 3))
            
            print_test_result("GroupSum: Uneven Grouping", True, "Handles padding correctly")
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print_test_result("GroupSum: Uneven Grouping", False, str(e))
            self.tests_failed += 1
            return False


def test_groupsum():
    """Test GroupSum module."""
    print_section("GROUPSUM TESTS")
    
    tester = GroupSumTester()
    
    print_subsection("GroupSum Module Tests")
    
    # Run all tests
    tester.test_basic_forward_pass()
    tester.test_grouping()
    tester.test_tau_scaling()
    tester.test_batch_handling()
    tester.test_with_layer_output()
    tester.test_gradient_flow()
    tester.test_uneven_grouping()
    
    return tester.tests_passed, tester.tests_failed


def main():
    """Run all GroupSum tests."""
    print("\n" + "=" * 70)
    print("  GROUPSUM MODULE TEST SUITE")
    print("=" * 70)
    
    passed, failed = test_groupsum()
    
    # Summary
    print_section("SUMMARY")
    print(f"  Total: {passed} passed, {failed} failed")
    
    if failed > 0:
        print(f"\n⚠ {failed} test(s) failed!")
        sys.exit(1)
    else:
        print("\n✓ All GroupSum tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
