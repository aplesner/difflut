"""
Integration tests: Nodes with Initializers
Tests that each node type works with each registered initializer.
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
@pytest.mark.parametrize("init_name", REGISTRY.list_initializers())
def test_node_with_initializer(node_name, init_name):
    """Test that each node works with each registered initializer."""
    node_class = REGISTRY.get_node(node_name)
    init_fn = REGISTRY.get_initializer(init_name)
    
    try:
        with IgnoreWarnings():
            # Create node with initializer
            node = instantiate_node(
                node_class,
                input_dim=4,
                output_dim=2,
                init_fn=init_fn
            )
        
        # Test forward pass works
        input_tensor = generate_uniform_input((8, 4), seed=42)
        with torch.no_grad():
            output = node(input_tensor)
        
        # Verify output shape and validity
        assert output.shape == (8, 2), (
            f"Node '{node_name}' with initializer '{init_name}' "
            f"produced wrong shape: {output.shape}"
        )
        assert not torch.isnan(output).any(), (
            f"Node '{node_name}' with initializer '{init_name}' produced NaN outputs"
        )
        assert not torch.isinf(output).any(), (
            f"Node '{node_name}' with initializer '{init_name}' produced Inf outputs"
        )
        
        # Verify parameters were initialized properly
        for param_name, param in node.named_parameters():
            assert not torch.isnan(param).any(), (
                f"Node '{node_name}' with initializer '{init_name}' "
                f"has NaN in parameter '{param_name}'"
            )
            assert not torch.isinf(param).any(), (
                f"Node '{node_name}' with initializer '{init_name}' "
                f"has Inf in parameter '{param_name}'"
            )
        
    except Exception as e:
        pytest.fail(
            f"Node '{node_name}' failed with initializer '{init_name}': {e}"
        )
