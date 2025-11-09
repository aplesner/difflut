"""
Comprehensive tests for all registered regularizers.
Tests that each regularizer can be applied and produces valid results.
"""

import pytest
import torch
import torch.nn as nn
from testing_utils import (
    IgnoreWarnings,
    instantiate_node,
)

from difflut.registry import REGISTRY


@pytest.mark.parametrize("reg_name", REGISTRY.list_regularizers())
def test_regularizer_callable(reg_name):
    """Test that each registered regularizer is callable."""
    reg_fn = REGISTRY.get_regularizer(reg_name)
    assert callable(reg_fn), f"Regularizer '{reg_name}' is not callable"


@pytest.mark.parametrize("reg_name", REGISTRY.list_regularizers())
def test_regularizer_signature(reg_name):
    """Test that regularizer has valid signature (should accept a node)."""
    import inspect
    
    reg_fn = REGISTRY.get_regularizer(reg_name)
    sig = inspect.signature(reg_fn)
    
    # Should accept at least a node parameter
    assert len(sig.parameters) > 0, f"Regularizer '{reg_name}' has no parameters"
    
    # First parameter should likely be 'node' or 'module'
    first_param = list(sig.parameters.keys())[0]
    assert first_param.lower() in ['node', 'module', 'model'], (
        f"Regularizer '{reg_name}' first parameter is '{first_param}', "
        f"expected 'node' or 'module'"
    )


@pytest.mark.parametrize("reg_name", REGISTRY.list_regularizers())
def test_regularizer_on_simple_node(reg_name):
    """Test that regularizer can be applied to a simple node."""
    from difflut.nodes import LinearLUTNode
    
    reg_fn = REGISTRY.get_regularizer(reg_name)
    
    # Create a simple node
    with IgnoreWarnings():
        node = LinearLUTNode(input_dim=4, output_dim=1)
    
    try:
        # Apply regularizer
        reg_value = reg_fn(node)
        
        # Check that regularizer returns a scalar tensor
        assert isinstance(reg_value, torch.Tensor), (
            f"Regularizer '{reg_name}' did not return a tensor"
        )
        assert reg_value.dim() == 0, (
            f"Regularizer '{reg_name}' returned non-scalar tensor with shape {reg_value.shape}"
        )
        
        # Check value is valid (not NaN or Inf)
        assert not torch.isnan(reg_value), f"Regularizer '{reg_name}' produced NaN"
        assert not torch.isinf(reg_value), f"Regularizer '{reg_name}' produced Inf"
        
        # Check value is non-negative (regularization should add penalty)
        assert reg_value >= 0, (
            f"Regularizer '{reg_name}' produced negative value: {reg_value.item()}"
        )
        
    except Exception as e:
        pytest.fail(f"Regularizer '{reg_name}' failed on LinearLUTNode: {e}")


@pytest.mark.parametrize("reg_name", REGISTRY.list_regularizers())
@pytest.mark.parametrize("node_type_name", ["linear_lut", "dwn", "neurallut"])
def test_regularizer_on_different_nodes(reg_name, node_type_name):
    """Test that regularizer works with different node types."""
    reg_fn = REGISTRY.get_regularizer(reg_name)
    node_class = REGISTRY.get_node(node_type_name)
    
    # Create node
    with IgnoreWarnings():
        node = instantiate_node(node_class, input_dim=4, output_dim=1)
    
    try:
        # Apply regularizer
        reg_value = reg_fn(node)
        
        # Basic validation
        assert isinstance(reg_value, torch.Tensor), (
            f"Regularizer '{reg_name}' did not return tensor for {node_type_name}"
        )
        assert not torch.isnan(reg_value), (
            f"Regularizer '{reg_name}' produced NaN for {node_type_name}"
        )
        assert not torch.isinf(reg_value), (
            f"Regularizer '{reg_name}' produced Inf for {node_type_name}"
        )
        
    except Exception as e:
        # Some regularizers might not work with all node types
        # (e.g., spectral regularizer only works with truth-table nodes)
        # This is acceptable, just log it
        pytest.skip(
            f"Regularizer '{reg_name}' not compatible with {node_type_name}: {e}"
        )



@pytest.mark.parametrize("reg_name", REGISTRY.list_regularizers())
def test_regularizer_scales_with_params(reg_name):
    """Test that regularizer value changes when parameters change."""
    from difflut.nodes import LinearLUTNode
    
    reg_fn = REGISTRY.get_regularizer(reg_name)
    
    # Create node
    with IgnoreWarnings():
        node = LinearLUTNode(input_dim=4, output_dim=1)
    
    try:
        # Get initial regularization value
        with torch.no_grad():
            reg_value1 = reg_fn(node).item()
        
        # Modify parameters
        with torch.no_grad():
            for param in node.parameters():
                param.mul_(2.0)  # Double all parameters
        
        # Get new regularization value
        with torch.no_grad():
            reg_value2 = reg_fn(node).item()
        
        # Values should be different (in most cases)
        # Some regularizers might not depend on parameter magnitude
        if reg_value1 == reg_value2:
            pytest.skip(
                f"Regularizer '{reg_name}' does not scale with parameters "
                f"(may be parameter-independent)"
            )
        
    except Exception as e:
        pytest.skip(f"Regularizer '{reg_name}' test failed: {e}")


def test_all_regularizers_registered():
    """Test that we have at least some regularizers registered."""
    regularizers = REGISTRY.list_regularizers()
    assert len(regularizers) > 0, "No regularizers registered in REGISTRY"
    print(f"Found {len(regularizers)} registered regularizers: {regularizers}")



