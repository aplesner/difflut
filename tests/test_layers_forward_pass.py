"""
Comprehensive forward pass tests for all layer types.
Tests: all nodes, shape correctness, output range [0,1], CPU/GPU consistency, gradients.
"""

import sys
import torch
import torch.nn as nn
from test_utils import (
    print_section,
    print_subsection,
    print_test_result,
    get_available_devices,
    is_cuda_available,
    get_all_registered_layers,
    get_all_registered_nodes,
    instantiate_layer,
    generate_uniform_input,
    assert_shape_equal,
    assert_range,
    assert_gradients_exist,
    compare_cpu_gpu_forward,
    IgnoreWarnings,
    CPU_GPU_ATOL,
    CPU_GPU_RTOL,
)


class LayerTester:
    """Helper class for testing layers."""
    
    def __init__(self, layer_name: str, layer_class: type):
        self.layer_name = layer_name
        self.layer_class = layer_class
        self.tests_passed = 0
        self.tests_failed = 0
    
    def test_with_all_nodes(self):
        """Test 2.1: Layer works with all node types."""
        nodes = get_all_registered_nodes()
        passed_nodes = 0
        failed_nodes = []
        
        for node_name, node_class in nodes.items():
            try:
                with IgnoreWarnings():
                    layer = instantiate_layer(
                        self.layer_class,
                        input_size=64,
                        output_size=32,
                        node_type=node_class,
                        n=4
                    )
                
                # Test forward pass
                input_tensor = generate_uniform_input((4, 64), seed=42)
                with torch.no_grad():
                    output = layer(input_tensor)
                
                assert output.shape[0] == 4  # Batch size preserved
                passed_nodes += 1
                
            except Exception as e:
                failed_nodes.append((node_name, str(e)))
        
        if failed_nodes:
            msg = f"Failed with {len(failed_nodes)} nodes: {[n for n, _ in failed_nodes]}"
            print_test_result(
                f"{self.layer_name}: All Nodes",
                len(failed_nodes) == 0,
                msg
            )
        else:
            print_test_result(
                f"{self.layer_name}: All Nodes",
                True,
                f"Tested with {passed_nodes} nodes"
            )
        
        if len(failed_nodes) == 0:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
        
        return len(failed_nodes) == 0
    
    def test_shape_forward_pass(self):
        """Test 2.2: Forward pass produces correct output shape."""
        try:
            with IgnoreWarnings():
                layer = instantiate_layer(
                    self.layer_class,
                    input_size=256,
                    output_size=128,
                    n=4
                )
            
            # Input shape: (batch_size, input_size)
            batch_size = 8
            input_tensor = generate_uniform_input((batch_size, 256))
            
            with torch.no_grad():
                output = layer(input_tensor)
            
            # Output shape should be: (batch_size, output_size)
            expected_shape = (batch_size, 128)
            assert_shape_equal(output, expected_shape)
            
            print_test_result(f"{self.layer_name}: Shape", True)
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print_test_result(f"{self.layer_name}: Shape", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_output_range(self):
        """Test 2.2: Output range is [0, 1]."""
        try:
            with IgnoreWarnings():
                layer = instantiate_layer(
                    self.layer_class,
                    input_size=256,
                    output_size=128,
                    n=4
                )
            layer.eval()
            
            # Test multiple random inputs
            for seed in [42, 123, 456]:
                input_tensor = generate_uniform_input((4, 256), seed=seed)
                
                with torch.no_grad():
                    output = layer(input_tensor)
                
                assert_range(output, 0.0, 1.0, msg=f"seed={seed}")
            
            print_test_result(f"{self.layer_name}: Output Range [0,1]", True)
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print_test_result(f"{self.layer_name}: Output Range [0,1]", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_cpu_gpu_consistency(self):
        """Test 2.2: CPU and GPU implementations give same forward pass."""
        if not is_cuda_available():
            print_test_result(f"{self.layer_name}: CPU/GPU Consistency", None, "CUDA not available")
            return None
        
        try:
            with IgnoreWarnings():
                layer = instantiate_layer(
                    self.layer_class,
                    input_size=256,
                    output_size=128,
                    n=4
                )
            layer.eval()
            
            input_tensor = generate_uniform_input((4, 256), seed=42)
            
            output_cpu, output_gpu = compare_cpu_gpu_forward(
                layer, input_tensor,
                atol=CPU_GPU_ATOL,
                rtol=CPU_GPU_RTOL
            )
            
            print_test_result(f"{self.layer_name}: CPU/GPU Consistency", True)
            self.tests_passed += 1
            return True
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                print_test_result(f"{self.layer_name}: CPU/GPU Consistency", None, "CUDA not available")
                return None
            print_test_result(f"{self.layer_name}: CPU/GPU Consistency", False, str(e))
            self.tests_failed += 1
            return False
        except Exception as e:
            print_test_result(f"{self.layer_name}: CPU/GPU Consistency", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_gradients(self):
        """Test 2.3: Gradients exist and are not all zero."""
        try:
            with IgnoreWarnings():
                layer = instantiate_layer(
                    self.layer_class,
                    input_size=256,
                    output_size=128,
                    n=4
                )
            layer.train()
            
            input_tensor = generate_uniform_input((4, 256), seed=42, device='cpu')
            input_tensor.requires_grad = True
            
            output = layer(input_tensor)
            loss = output.sum()
            loss.backward()
            
            # Check that gradients exist and are not all zero
            assert_gradients_exist(layer, msg="Layer has zero gradients")
            
            print_test_result(f"{self.layer_name}: Gradients", True)
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print_test_result(f"{self.layer_name}: Gradients", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_different_layer_sizes(self):
        """Test layer with different input/output sizes."""
        try:
            test_configs = [
                (100, 50),
                (512, 256),
                (1000, 100),
            ]
            
            for input_size, output_size in test_configs:
                with IgnoreWarnings():
                    layer = instantiate_layer(
                        self.layer_class,
                        input_size=input_size,
                        output_size=output_size,
                        n=4
                    )
                
                input_tensor = generate_uniform_input((2, input_size), seed=42)
                
                with torch.no_grad():
                    output = layer(input_tensor)
                
                assert_shape_equal(output, (2, output_size))
            
            print_test_result(f"{self.layer_name}: Different Sizes", True, f"Tested {len(test_configs)} configs")
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print_test_result(f"{self.layer_name}: Different Sizes", False, str(e))
            self.tests_failed += 1
            return False


def test_all_layers():
    """Test all registered layers."""
    print_section("LAYER FORWARD PASS TESTS")
    
    layers = get_all_registered_layers()
    print(f"\nTesting {len(layers)} layers: {list(layers.keys())}\n")
    
    all_passed = 0
    all_failed = 0
    
    for layer_name, layer_class in layers.items():
        print_subsection(f"Layer: {layer_name}")
        
        tester = LayerTester(layer_name, layer_class)
        
        # Run all tests
        tester.test_with_all_nodes()
        tester.test_shape_forward_pass()
        tester.test_output_range()
        tester.test_cpu_gpu_consistency()
        tester.test_gradients()
        tester.test_different_layer_sizes()
        
        all_passed += tester.tests_passed
        all_failed += tester.tests_failed
        
        print(f"  → {tester.tests_passed} passed, {tester.tests_failed} failed")
    
    return all_passed, all_failed


def main():
    """Run all layer tests."""
    print("\n" + "=" * 70)
    print("  LAYER FORWARD PASS TEST SUITE")
    print("=" * 70)
    
    passed, failed = test_all_layers()
    
    # Summary
    print_section("SUMMARY")
    print(f"  Total: {passed} passed, {failed} failed")
    
    if failed > 0:
        print(f"\n⚠ {failed} test(s) failed!")
        sys.exit(1)
    else:
        print("\n✓ All layer tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
