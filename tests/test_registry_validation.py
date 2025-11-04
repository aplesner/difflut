"""
Registry validation tests.
Ensures all registered components are actually implemented and can be instantiated.
"""

import sys
import traceback
from test_utils import (
    print_section,
    print_subsection,
    print_test_result,
    get_all_registered_nodes,
    get_all_registered_layers,
    get_all_registered_encoders,
    get_all_registered_initializers,
    get_all_registered_regularizers,
    instantiate_node,
    instantiate_layer,
    instantiate_encoder,
    IgnoreWarnings
)


def test_registry_nodes_implemented():
    """Test that all registered nodes are actually implemented."""
    print_section("REGISTRY VALIDATION: Nodes")
    
    nodes = get_all_registered_nodes()
    print(f"\nFound {len(nodes)} registered nodes: {list(nodes.keys())}")
    
    passed = 0
    failed = 0
    
    for node_name, node_class in nodes.items():
        try:
            # Try to instantiate the node
            with IgnoreWarnings():
                node = instantiate_node(node_class, input_dim=4, output_dim=1, layer_size=2)
            
            # Check that it's a proper PyTorch module
            import torch.nn as nn
            assert isinstance(node, nn.Module), f"Node {node_name} is not an nn.Module"
            
            # Check that it has forward method
            assert hasattr(node, 'forward'), f"Node {node_name} has no forward method"
            
            print_test_result(node_name, True)
            passed += 1
            
        except Exception as e:
            print_test_result(node_name, False, str(e))
            failed += 1
    
    print(f"\nNodes: {passed} passed, {failed} failed")
    return failed == 0


def test_registry_layers_implemented():
    """Test that all registered layers are actually implemented."""
    print_section("REGISTRY VALIDATION: Layers")
    
    layers = get_all_registered_layers()
    print(f"\nFound {len(layers)} registered layers: {list(layers.keys())}")
    
    passed = 0
    failed = 0
    
    for layer_name, layer_class in layers.items():
        try:
            # Try to instantiate the layer
            with IgnoreWarnings():
                layer = instantiate_layer(layer_class, input_size=256, output_size=128, n=4)
            
            # Check that it's a proper PyTorch module
            import torch.nn as nn
            assert isinstance(layer, nn.Module), f"Layer {layer_name} is not an nn.Module"
            
            # Check that it has forward method
            assert hasattr(layer, 'forward'), f"Layer {layer_name} has no forward method"
            
            print_test_result(layer_name, True)
            passed += 1
            
        except Exception as e:
            print_test_result(layer_name, False, str(e))
            failed += 1
    
    print(f"\nLayers: {passed} passed, {failed} failed")
    return failed == 0


def test_registry_encoders_implemented():
    """Test that all registered encoders are actually implemented."""
    print_section("REGISTRY VALIDATION: Encoders")
    
    encoders = get_all_registered_encoders()
    print(f"\nFound {len(encoders)} registered encoders: {list(encoders.keys())}")
    
    passed = 0
    failed = 0
    
    for encoder_name, encoder_class in encoders.items():
        try:
            # Try to instantiate the encoder
            with IgnoreWarnings():
                encoder = instantiate_encoder(encoder_class, num_bits=8)
            
            # Check that it's a proper PyTorch module
            import torch.nn as nn
            assert isinstance(encoder, nn.Module), f"Encoder {encoder_name} is not an nn.Module"
            
            # Check that it has forward method
            assert hasattr(encoder, 'forward'), f"Encoder {encoder_name} has no forward method"
            
            print_test_result(encoder_name, True)
            passed += 1
            
        except Exception as e:
            print_test_result(encoder_name, False, str(e))
            failed += 1
    
    print(f"\nEncoders: {passed} passed, {failed} failed")
    return failed == 0


def test_registry_initializers_implemented():
    """Test that all registered initializers are actually implemented."""
    print_section("REGISTRY VALIDATION: Initializers")
    
    initializers = get_all_registered_initializers()
    print(f"\nFound {len(initializers)} registered initializers: {list(initializers.keys())}")
    
    if len(initializers) == 0:
        print("  No initializers registered (this is okay)")
        return True
    
    passed = 0
    failed = 0
    
    for init_name, init_func in initializers.items():
        try:
            # Check that it's callable
            assert callable(init_func), f"Initializer {init_name} is not callable"
            
            print_test_result(init_name, True)
            passed += 1
            
        except Exception as e:
            print_test_result(init_name, False, str(e))
            failed += 1
    
    print(f"\nInitializers: {passed} passed, {failed} failed")
    return failed == 0


def test_registry_regularizers_implemented():
    """Test that all registered regularizers are actually implemented."""
    print_section("REGISTRY VALIDATION: Regularizers")
    
    regularizers = get_all_registered_regularizers()
    print(f"\nFound {len(regularizers)} registered regularizers: {list(regularizers.keys())}")
    
    if len(regularizers) == 0:
        print("  No regularizers registered (this is okay)")
        return True
    
    passed = 0
    failed = 0
    
    for reg_name, reg_func in regularizers.items():
        try:
            # Check that it's callable
            assert callable(reg_func), f"Regularizer {reg_name} is not callable"
            
            print_test_result(reg_name, True)
            passed += 1
            
        except Exception as e:
            print_test_result(reg_name, False, str(e))
            failed += 1
    
    print(f"\nRegularizers: {passed} passed, {failed} failed")
    return failed == 0


def test_registry_consistency():
    """Test that registry methods are consistent."""
    print_section("REGISTRY VALIDATION: Consistency")
    
    from difflut.registry import REGISTRY
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: list_all() returns expected structure
    try:
        all_components = REGISTRY.list_all()
        expected_keys = {'nodes', 'layers', 'encoders', 'initializers', 'regularizers'}
        assert set(all_components.keys()) == expected_keys, "list_all() missing expected keys"
        print_test_result("list_all() structure", True)
        tests_passed += 1
    except Exception as e:
        print_test_result("list_all() structure", False, str(e))
        tests_failed += 1
    
    # Test 2: list_nodes() == list_all()['nodes']
    try:
        nodes_direct = REGISTRY.list_nodes()
        nodes_from_all = REGISTRY.list_all()['nodes']
        assert nodes_direct == nodes_from_all, "list_nodes() inconsistent with list_all()"
        print_test_result("list_nodes() consistency", True)
        tests_passed += 1
    except Exception as e:
        print_test_result("list_nodes() consistency", False, str(e))
        tests_failed += 1
    
    # Test 3: get_node() works for all listed nodes
    try:
        for node_name in REGISTRY.list_nodes():
            node_class = REGISTRY.get_node(node_name)
            assert node_class is not None
        print_test_result("get_node() for all listed", True, f"({len(REGISTRY.list_nodes())} nodes)")
        tests_passed += 1
    except Exception as e:
        print_test_result("get_node() for all listed", False, str(e))
        tests_failed += 1
    
    # Test 4: Invalid component names raise ValueError
    try:
        try:
            REGISTRY.get_node('nonexistent_node')
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass  # Expected
        
        try:
            REGISTRY.get_layer('nonexistent_layer')
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass  # Expected
        
        print_test_result("Invalid names raise ValueError", True)
        tests_passed += 1
    except Exception as e:
        print_test_result("Invalid names raise ValueError", False, str(e))
        tests_failed += 1
    
    print(f"\nConsistency: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def main():
    """Run all registry validation tests."""
    print("\n" + "=" * 70)
    print("  REGISTRY VALIDATION TEST SUITE")
    print("=" * 70)
    
    results = {
        'Nodes': test_registry_nodes_implemented(),
        'Layers': test_registry_layers_implemented(),
        'Encoders': test_registry_encoders_implemented(),
        'Initializers': test_registry_initializers_implemented(),
        'Regularizers': test_registry_regularizers_implemented(),
        'Consistency': test_registry_consistency(),
    }
    
    # Summary
    print_section("SUMMARY")
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{len(results)} test groups passed")
    
    if failed > 0:
        print(f"\n⚠ {failed} test group(s) failed!")
        sys.exit(1)
    else:
        print("\n✓ All registry validation tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
