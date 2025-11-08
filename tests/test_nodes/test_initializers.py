"""
Comprehensive tests for all registered initializers.
Tests that each initializer can be applied and produces valid results.
"""

import pytest
import torch
import torch.nn as nn
from testing_utils import (
    IgnoreWarnings,
    instantiate_node,
)

from difflut.registry import REGISTRY


@pytest.mark.parametrize("init_name", REGISTRY.list_initializers())
def test_initializer_callable(init_name):
    """Test that each registered initializer is callable."""
    init_fn = REGISTRY.get_initializer(init_name)
    assert callable(init_fn), f"Initializer '{init_name}' is not callable"


@pytest.mark.parametrize("init_name", REGISTRY.list_initializers())
def test_initializer_signature(init_name):
    """Test that initializer has valid signature."""
    import inspect
    
    init_fn = REGISTRY.get_initializer(init_name)
    sig = inspect.signature(init_fn)
    
    # Should accept at least a tensor parameter
    assert len(sig.parameters) > 0, f"Initializer '{init_name}' has no parameters"


@pytest.mark.parametrize("init_name", REGISTRY.list_initializers())
def test_initializer_on_tensor(init_name):
    """Test that initializer can be applied to a tensor."""
    init_fn = REGISTRY.get_initializer(init_name)
    
    # Create test tensors of different shapes
    test_shapes = [(10,), (10, 10), (64,), (16, 4)]
    
    for shape in test_shapes:
        tensor = torch.zeros(shape)
        
        try:
            # Try to apply initializer
            init_fn(tensor)
            
            # Check tensor was modified (not all zeros anymore, in most cases)
            # Some initializers might set to zero, so we just check it's still a valid tensor
            assert tensor.shape == shape, f"Initializer '{init_name}' changed tensor shape"
            assert not torch.isnan(tensor).any(), f"Initializer '{init_name}' produced NaN values"
            assert not torch.isinf(tensor).any(), f"Initializer '{init_name}' produced Inf values"
            
        except Exception as e:
            pytest.fail(f"Initializer '{init_name}' failed on shape {shape}: {e}")


@pytest.mark.parametrize("init_name", REGISTRY.list_initializers())
def test_initializer_with_node_params(init_name):
    """Test that initializer works with actual node parameters."""
    from difflut.nodes import LinearLUTNode
    
    init_fn = REGISTRY.get_initializer(init_name)
    
    # Create a simple node
    with IgnoreWarnings():
        node = LinearLUTNode(input_dim=4, output_dim=1)
    
    # Apply initializer to all parameters
    for name, param in node.named_parameters():
        if param.requires_grad:
            try:
                init_fn(param.data)
                
                # Verify parameter is still valid
                assert not torch.isnan(param).any(), (
                    f"Initializer '{init_name}' produced NaN in parameter '{name}'"
                )
                assert not torch.isinf(param).any(), (
                    f"Initializer '{init_name}' produced Inf in parameter '{name}'"
                )
                
            except Exception as e:
                pytest.fail(
                    f"Initializer '{init_name}' failed on parameter '{name}': {e}"
                )


@pytest.mark.parametrize("init_name", REGISTRY.list_initializers())
def test_initializer_deterministic(init_name):
    """Test that initializer produces consistent results with same seed."""
    init_fn = REGISTRY.get_initializer(init_name)
    
    # Create two identical tensors
    torch.manual_seed(42)
    tensor1 = torch.zeros(10, 10)
    init_fn(tensor1)
    
    torch.manual_seed(42)
    tensor2 = torch.zeros(10, 10)
    init_fn(tensor2)
    
    # Should produce same result with same seed
    assert torch.allclose(tensor1, tensor2, atol=1e-6), (
        f"Initializer '{init_name}' is not deterministic with same seed"
    )


@pytest.mark.parametrize("init_name", REGISTRY.list_initializers())
def test_initializer_range(init_name):
    """Test that initializer produces values in reasonable range."""
    init_fn = REGISTRY.get_initializer(init_name)
    
    # Create test tensor
    tensor = torch.zeros(1000)
    init_fn(tensor)
    
    # Check values are in a reasonable range (for LUT values should be close to [0,1])
    # Allow some flexibility as different initializers have different ranges
    assert tensor.min() >= -10.0, f"Initializer '{init_name}' produced values < -10"
    assert tensor.max() <= 10.0, f"Initializer '{init_name}' produced values > 10"


def test_all_initializers_registered():
    """Test that we have at least some initializers registered."""
    initializers = REGISTRY.list_initializers()
    assert len(initializers) > 0, "No initializers registered in REGISTRY"
    print(f"Found {len(initializers)} registered initializers: {initializers}")
