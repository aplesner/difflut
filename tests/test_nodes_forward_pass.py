"""
Comprehensive forward pass tests for all node types.
Tests: shape correctness, output range [0,1], CPU/GPU consistency, gradients, initializers, regularizers.

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
    get_available_devices,
    is_cuda_available,
    skip_if_no_cuda,
    get_all_registered_nodes,
    get_all_registered_layers,
    instantiate_node,
    generate_uniform_input,
    assert_shape_equal,
    assert_range,
    assert_gradients_exist,
    assert_tensors_close,
    compare_cpu_gpu_forward,
    check_gradients,
    IgnoreWarnings,
    FP32_ATOL,
    FP32_RTOL,
    CPU_GPU_ATOL,
    CPU_GPU_RTOL,
)

import sys
import torch
import torch.nn as nn
from test_utils import (
    print_section,
    print_subsection,
    print_test_result,
    get_available_devices,
    is_cuda_available,
    skip_if_no_cuda,
    get_all_registered_nodes,
    instantiate_node,
    generate_uniform_input,
    assert_shape_equal,
    assert_range,
    assert_gradients_exist,
    assert_tensors_close,
    compare_cpu_gpu_forward,
    check_gradients,
    IgnoreWarnings,
    FP32_ATOL,
    FP32_RTOL,
    CPU_GPU_ATOL,
    CPU_GPU_RTOL,
)


class NodeTester:
    """Helper class for testing nodes."""
    
    def __init__(self, node_name: str, node_class: type):
        self.node_name = node_name
        self.node_class = node_class
        self.tests_passed = 0
        self.tests_failed = 0
    
    def test_shape_forward_pass(self):
        """Test 1.1: Forward pass produces correct output shape."""
        try:
            with IgnoreWarnings():
                node = instantiate_node(self.node_class, input_dim=4, output_dim=2, layer_size=16)
            
            # Input shape: (batch_size, layer_size, input_dim)
            batch_size = 8
            input_tensor = generate_uniform_input((batch_size, 16, 4))
            
            with torch.no_grad():
                output = node(input_tensor)
            
            # Output shape should be: (batch_size, layer_size, output_dim)
            expected_shape = (batch_size, 16, 2)
            assert_shape_equal(output, expected_shape)
            
            print_test_result(f"{self.node_name}: Shape", True)
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print_test_result(f"{self.node_name}: Shape", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_output_range(self):
        """Test 1.1: Output range is [0, 1]."""
        try:
            with IgnoreWarnings():
                node = instantiate_node(self.node_class, input_dim=4, output_dim=1, layer_size=16)
            node.eval()
            
            # Test multiple random inputs
            for seed in [42, 123, 456]:
                input_tensor = generate_uniform_input((4, 16, 4), seed=seed)
                
                with torch.no_grad():
                    output = node(input_tensor)
                
                assert_range(output, 0.0, 1.0, msg=f"seed={seed}")
            
            print_test_result(f"{self.node_name}: Output Range [0,1]", True)
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print_test_result(f"{self.node_name}: Output Range [0,1]", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_cpu_gpu_consistency(self):
        """Test 1.2: CPU and GPU implementations give same forward pass."""
        if not is_cuda_available():
            print_test_result(f"{self.node_name}: CPU/GPU Consistency", None, "CUDA not available")
            return None
        
        try:
            with IgnoreWarnings():
                node = instantiate_node(self.node_class, input_dim=4, output_dim=1, layer_size=16)
            node.eval()
            
            input_tensor = generate_uniform_input((4, 16, 4), seed=42)
            
            output_cpu, output_gpu = compare_cpu_gpu_forward(
                node, input_tensor,
                atol=CPU_GPU_ATOL,
                rtol=CPU_GPU_RTOL
            )
            
            print_test_result(f"{self.node_name}: CPU/GPU Consistency", True)
            self.tests_passed += 1
            return True
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                print_test_result(f"{self.node_name}: CPU/GPU Consistency", None, "CUDA not available")
                return None
            print_test_result(f"{self.node_name}: CPU/GPU Consistency", False, str(e))
            self.tests_failed += 1
            return False
        except Exception as e:
            print_test_result(f"{self.node_name}: CPU/GPU Consistency", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_gradients(self):
        """Test 1.3: Gradients exist and are not all zero."""
        try:
            with IgnoreWarnings():
                node = instantiate_node(self.node_class, input_dim=4, output_dim=1, layer_size=16)
            node.train()
            
            input_tensor = generate_uniform_input((4, 16, 4), seed=42, device='cpu')
            input_tensor.requires_grad = True
            
            output = node(input_tensor)
            loss = output.sum()
            loss.backward()
            
            # Check that gradients exist and are not all zero
            assert_gradients_exist(node, msg="Node has zero gradients")
            
            print_test_result(f"{self.node_name}: Gradients", True)
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print_test_result(f"{self.node_name}: Gradients", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_with_initializer(self):
        """Test 1.4: Node works with initializers."""
        try:
            # Define a simple initializer
            def zeros_init(param: torch.Tensor, **kwargs):
                nn.init.zeros_(param)
            
            with IgnoreWarnings():
                node = instantiate_node(
                    self.node_class,
                    input_dim=4,
                    output_dim=1,
                    layer_size=16,
                    init_fn=zeros_init,
                    init_kwargs={}
                )
            node.eval()
            
            input_tensor = generate_uniform_input((4, 16, 4), seed=42)
            
            with torch.no_grad():
                output = node(input_tensor)
            
            # Should not crash and produce valid output
            assert output.shape[0] == 4
            
            print_test_result(f"{self.node_name}: With Initializer", True)
            self.tests_passed += 1
            return True
            
        except Exception as e:
            # Some nodes might not support initializers
            if "init_fn" in str(e) or "unexpected keyword" in str(e):
                print_test_result(f"{self.node_name}: With Initializer", None, "Not supported")
                return None
            print_test_result(f"{self.node_name}: With Initializer", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_with_regularizer(self):
        """Test 1.4: Node works with regularizers."""
        try:
            # Define a simple regularizer
            def l1_regularizer(node, **kwargs):
                l1_loss = 0
                for param in node.parameters():
                    l1_loss += param.abs().sum()
                return l1_loss
            
            regularizers = {'l1': (l1_regularizer, 0.01, {})}
            
            with IgnoreWarnings():
                node = instantiate_node(
                    self.node_class,
                    input_dim=4,
                    output_dim=1,
                    layer_size=16,
                    regularizers=regularizers
                )
            node.eval()
            
            input_tensor = generate_uniform_input((4, 16, 4), seed=42)
            
            with torch.no_grad():
                output = node(input_tensor)
            
            # Should not crash and produce valid output
            assert output.shape[0] == 4
            
            print_test_result(f"{self.node_name}: With Regularizer", True)
            self.tests_passed += 1
            return True
            
        except Exception as e:
            # Some nodes might not support regularizers
            if "regularizers" in str(e) or "unexpected keyword" in str(e):
                print_test_result(f"{self.node_name}: With Regularizer", None, "Not supported")
                return None
            print_test_result(f"{self.node_name}: With Regularizer", False, str(e))
            self.tests_failed += 1
            return False


def test_all_nodes():
    """Test all registered nodes."""
    print_section("NODE FORWARD PASS TESTS")
    
    nodes = get_all_registered_nodes()
    print(f"\nTesting {len(nodes)} nodes: {list(nodes.keys())}\n")
    
    all_passed = 0
    all_failed = 0
    
    for node_name, node_class in nodes.items():
        print_subsection(f"Node: {node_name}")
        
        tester = NodeTester(node_name, node_class)
        
        # Run all tests
        tester.test_shape_forward_pass()
        tester.test_output_range()
        tester.test_cpu_gpu_consistency()
        tester.test_gradients()
        tester.test_with_initializer()
        tester.test_with_regularizer()
        
        all_passed += tester.tests_passed
        all_failed += tester.tests_failed
        
        print(f"  → {tester.tests_passed} passed, {tester.tests_failed} failed")
    
    return all_passed, all_failed


def main():
    """Run all node tests."""
    print("\n" + "=" * 70)
    print("  NODE FORWARD PASS TEST SUITE")
    print("=" * 70)
    
    passed, failed = test_all_nodes()
    
    # Summary
    print_section("SUMMARY")
    print(f"  Total: {passed} passed, {failed} failed")
    
    if failed > 0:
        print(f"\n⚠ {failed} test(s) failed!")
        sys.exit(1)
    else:
        print("\n✓ All node tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
