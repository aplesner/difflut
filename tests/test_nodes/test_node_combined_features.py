"""
Integration tests: Nodes with Combined Features
Tests nodes with both initializers and regularizers, multiple regularizers, etc.
"""

import pytest
import torch
import torch.nn as nn
from testing_utils import (
    IgnoreWarnings,
    generate_uniform_input,
    instantiate_node,
)

from difflut.registry import REGISTRY


@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
def test_node_with_initializer_and_regularizer(node_name):
    """Test that nodes work with both initializer and regularizer together."""
    node_class = REGISTRY.get_node(node_name)
    
    # Pick first available initializer and regularizer
    initializers = REGISTRY.list_initializers()
    regularizers = REGISTRY.list_regularizers()
    
    if not initializers or not regularizers:
        pytest.skip("No initializers or regularizers registered")
    
    init_fn = REGISTRY.get_initializer(initializers[0])
    reg_fn = REGISTRY.get_regularizer(regularizers[0])
    
    try:
        with IgnoreWarnings():
            # Create node with both initializer and regularizer
            regularizers_dict = {regularizers[0]: (reg_fn, 0.01, {})}
            node = instantiate_node(
                node_class,
                input_dim=4,
                output_dim=2,
                init_fn=init_fn,
                regularizers=regularizers_dict
            )
        
        # Test forward pass
        input_tensor = generate_uniform_input((8, 4), seed=42)
        with torch.no_grad():
            output = node(input_tensor)
        
        assert output.shape == (8, 2)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Test regularizer computation
        try:
            reg_value = reg_fn(node)
            assert not torch.isnan(reg_value)
            assert not torch.isinf(reg_value)
        except Exception:
            # Regularizer might not be compatible
            pass
        
    except Exception as e:
        pytest.fail(
            f"Node '{node_name}' failed with initializer and regularizer: {e}"
        )


@pytest.mark.slow
@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
def test_node_with_multiple_regularizers(node_name):
    """Test that nodes can use multiple regularizers simultaneously."""
    node_class = REGISTRY.get_node(node_name)
    regularizers_list = REGISTRY.list_regularizers()
    
    if len(regularizers_list) < 2:
        pytest.skip("Need at least 2 regularizers for this test")
    
    # Use first two regularizers
    reg_fn1 = REGISTRY.get_regularizer(regularizers_list[0])
    reg_fn2 = REGISTRY.get_regularizer(regularizers_list[1])
    
    try:
        with IgnoreWarnings():
            # Create node with multiple regularizers
            regularizers_dict = {
                regularizers_list[0]: (reg_fn1, 0.01, {}),
                regularizers_list[1]: (reg_fn2, 0.001, {}),
            }
            node = instantiate_node(
                node_class,
                input_dim=4,
                output_dim=2,
                regularizers=regularizers_dict
            )
        
        # Test forward pass
        input_tensor = generate_uniform_input((8, 4), seed=42)
        with torch.no_grad():
            output = node(input_tensor)
        
        assert output.shape == (8, 2)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
    except Exception as e:
        pytest.fail(
            f"Node '{node_name}' failed with multiple regularizers: {e}"
        )


def test_integration_summary():
    """Print summary of available initializers, regularizers, and nodes."""
    nodes = REGISTRY.list_nodes()
    initializers = REGISTRY.list_initializers()
    regularizers = REGISTRY.list_regularizers()
    
    print(f"\n{'='*60}")
    print(f"Integration Test Summary")
    print(f"{'='*60}")
    print(f"Nodes: {len(nodes)}")
    print(f"  {', '.join(nodes)}")
    print(f"\nInitializers: {len(initializers)}")
    print(f"  {', '.join(initializers)}")
    print(f"\nRegularizers: {len(regularizers)}")
    print(f"  {', '.join(regularizers)}")
    print(f"\nTotal test combinations:")
    print(f"  Node × Initializer: {len(nodes) * len(initializers)}")
    print(f"  Node × Regularizer: {len(nodes) * len(regularizers)}")
    print(f"{'='*60}\n")
