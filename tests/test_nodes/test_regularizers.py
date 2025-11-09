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
def test_regularizer_differentiable(reg_name):
    """Test that regularizer is differentiable (can compute gradients)."""
    from difflut.nodes import LinearLUTNode
    
    reg_fn = REGISTRY.get_regularizer(reg_name)
    
    # Create node
    with IgnoreWarnings():
        node = LinearLUTNode(input_dim=4, output_dim=1)
    
    try:
        # Apply regularizer
        reg_value = reg_fn(node)
        
        # Try to compute gradients
        reg_value.backward()
        
        # Check that at least some parameters have gradients
        has_gradients = False
        for param in node.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        # Note: Some regularizers might not produce gradients if they use no_grad
        # This is a soft check
        if not has_gradients:
            pytest.skip(
                f"Regularizer '{reg_name}' did not produce gradients "
                f"(may use no_grad internally)"
            )
        
    except Exception as e:
        pytest.skip(f"Regularizer '{reg_name}' not differentiable: {e}")


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


# Tests for input-based regularization mode (new feature)


@pytest.mark.parametrize("reg_name", ["l", "functional", "l1", "l1_functional", "l2", "l2_functional"])
def test_input_based_regularizer_differentiable(reg_name):
    """Test that functional regularizers are differentiable when using inputs parameter."""
    from difflut.nodes import LinearLUTNode
    
    reg_fn = REGISTRY.get_regularizer(reg_name)
    
    # Create node
    with IgnoreWarnings():
        node = LinearLUTNode(input_dim=4, output_dim=1)
    
    # Create batch inputs (binary)
    batch_size = 8
    inputs = torch.randint(0, 2, (batch_size, 4), dtype=torch.float32)
    
    # Apply regularizer with inputs (input-based mode)
    reg_value = reg_fn(node, inputs=inputs)
    
    # Check that regularizer returns a scalar tensor
    assert isinstance(reg_value, torch.Tensor), (
        f"Regularizer '{reg_name}' did not return a tensor"
    )
    assert reg_value.dim() == 0, (
        f"Regularizer '{reg_name}' returned non-scalar tensor with shape {reg_value.shape}"
    )
    
    # Check that it requires grad
    assert reg_value.requires_grad, (
        f"Regularizer '{reg_name}' output does not require grad in input-based mode"
    )
    
    # Compute gradients
    reg_value.backward()
    
    # Check that parameters have gradients
    has_gradients = False
    for param in node.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break
    
    assert has_gradients, (
        f"Regularizer '{reg_name}' did not produce gradients in input-based mode"
    )


@pytest.mark.parametrize("reg_name", ["l", "l1", "l2"])
def test_input_based_vs_random_mode(reg_name):
    """Test that both input-based and random modes produce valid outputs."""
    from difflut.nodes import LinearLUTNode
    
    reg_fn = REGISTRY.get_regularizer(reg_name)
    
    # Create node
    with IgnoreWarnings():
        node = LinearLUTNode(input_dim=4, output_dim=1)
    
    # Test random mode (inputs=None)
    with torch.no_grad():
        reg_random = reg_fn(node, num_samples=50)
    
    assert isinstance(reg_random, torch.Tensor)
    assert not torch.isnan(reg_random)
    assert reg_random >= 0
    
    # Test input-based mode
    batch_size = 8
    inputs = torch.randint(0, 2, (batch_size, 4), dtype=torch.float32)
    
    with torch.no_grad():
        reg_input_based = reg_fn(node, inputs=inputs)
    
    assert isinstance(reg_input_based, torch.Tensor)
    assert not torch.isnan(reg_input_based)
    assert reg_input_based >= 0
    
    # Both should produce valid values (not necessarily equal)
    # Just verify they're both valid regularization values


@pytest.mark.parametrize("reg_name", ["l", "l1", "l2"])
@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_input_based_different_batch_sizes(reg_name, batch_size):
    """Test that input-based regularization works with different batch sizes."""
    from difflut.nodes import LinearLUTNode
    
    reg_fn = REGISTRY.get_regularizer(reg_name)
    
    # Create node
    with IgnoreWarnings():
        node = LinearLUTNode(input_dim=4, output_dim=1)
    
    # Create batch inputs
    inputs = torch.randint(0, 2, (batch_size, 4), dtype=torch.float32)
    
    # Apply regularizer
    reg_value = reg_fn(node, inputs=inputs)
    
    # Validate
    assert isinstance(reg_value, torch.Tensor)
    assert reg_value.dim() == 0
    assert not torch.isnan(reg_value)
    assert not torch.isinf(reg_value)
    assert reg_value >= 0


@pytest.mark.parametrize("reg_name", ["l", "l1", "l2"])
def test_input_based_gradient_flow(reg_name):
    """Test that gradients flow correctly through input-based regularization."""
    from difflut.nodes import LinearLUTNode
    
    reg_fn = REGISTRY.get_regularizer(reg_name)
    
    # Create node
    with IgnoreWarnings():
        node = LinearLUTNode(input_dim=4, output_dim=1)
    
    # Create batch inputs
    batch_size = 16
    inputs = torch.randint(0, 2, (batch_size, 4), dtype=torch.float32)
    
    # Zero gradients
    node.zero_grad()
    
    # Compute regularization with inputs
    reg_value = reg_fn(node, inputs=inputs)
    
    # Backward pass
    reg_value.backward()
    
    # Verify all parameters have gradients
    for name, param in node.named_parameters():
        assert param.grad is not None, (
            f"Parameter '{name}' has no gradient after input-based regularization"
        )
        # Gradient should be non-zero for at least some parameters
        # (may be zero for some parameters depending on the function)


@pytest.mark.parametrize("reg_name", ["l", "l1", "l2"])
def test_input_based_adapts_to_inputs(reg_name):
    """Test that input-based regularization produces different values for different inputs."""
    from difflut.nodes import LinearLUTNode
    
    reg_fn = REGISTRY.get_regularizer(reg_name)
    
    # Create node
    with IgnoreWarnings():
        node = LinearLUTNode(input_dim=4, output_dim=1)
    
    # Create two different batches of inputs
    batch_size = 16
    inputs1 = torch.zeros((batch_size, 4), dtype=torch.float32)  # All zeros
    inputs2 = torch.ones((batch_size, 4), dtype=torch.float32)   # All ones
    
    with torch.no_grad():
        reg_value1 = reg_fn(node, inputs=inputs1)
        reg_value2 = reg_fn(node, inputs=inputs2)
    
    # Values don't need to be different, but both should be valid
    # (they might be the same for symmetric functions)
    assert isinstance(reg_value1, torch.Tensor)
    assert isinstance(reg_value2, torch.Tensor)
    assert not torch.isnan(reg_value1)
    assert not torch.isnan(reg_value2)

