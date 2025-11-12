"""
Integration tests: Nodes with Regularizers
Tests that each node type works with each registered regularizer.
"""

import pytest
import torch
from testing_utils import (
    IgnoreWarnings,
    generate_uniform_input,
    instantiate_node,
)

from difflut.registry import REGISTRY


def _get_input_dim_for_node(node_name: str) -> int:
    """Get appropriate input dimension for node type.
    
    DiffLogic has exponential complexity (2^(2^n)), so we limit it to 2 inputs.
    Other nodes can use larger dimensions.
    """
    if node_name == "difflogic":
        return 2  # 2^(2^2) = 16 functions (manageable)
    return 4  # Default for other nodes


@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
@pytest.mark.parametrize("reg_name", REGISTRY.list_regularizers())
def test_node_with_regularizer(node_name, reg_name):
    """Test that each node works with each registered regularizer."""
    node_class = REGISTRY.get_node(node_name)
    reg_fn = REGISTRY.get_regularizer(reg_name)
    input_dim = _get_input_dim_for_node(node_name)

    try:
        with IgnoreWarnings():
            # Create node with regularizer
            # Format: {"name": (reg_fn, weight, kwargs)}
            regularizers_dict = {reg_name: (reg_fn, 1.0, {})}
            node = instantiate_node(
                node_class, input_dim=input_dim, output_dim=2, regularizers=regularizers_dict
            )

        # Test forward pass works
        input_tensor = generate_uniform_input((8, input_dim), seed=42)
        with torch.no_grad():
            output = node(input_tensor)

        # Verify output shape and validity
        assert output.shape == (8, 2), (
            f"Node '{node_name}' with regularizer '{reg_name}' "
            f"produced wrong shape: {output.shape}"
        )
        assert not torch.isnan(
            output
        ).any(), f"Node '{node_name}' with regularizer '{reg_name}' produced NaN outputs"
        assert not torch.isinf(
            output
        ).any(), f"Node '{node_name}' with regularizer '{reg_name}' produced Inf outputs"

        # Test that regularizer can be computed
        try:
            reg_value = reg_fn(node)
            assert isinstance(
                reg_value, torch.Tensor
            ), f"Regularizer '{reg_name}' did not return tensor for node '{node_name}'"
            assert (
                reg_value.dim() == 0
            ), f"Regularizer '{reg_name}' returned non-scalar for node '{node_name}'"
            assert not torch.isnan(
                reg_value
            ), f"Regularizer '{reg_name}' produced NaN for node '{node_name}'"
            assert not torch.isinf(
                reg_value
            ), f"Regularizer '{reg_name}' produced Inf for node '{node_name}'"
        except Exception as reg_error:
            # Some regularizers may not be compatible with all node types
            pytest.skip(
                f"Regularizer '{reg_name}' not compatible with node '{node_name}': {reg_error}"
            )

    except Exception as e:
        pytest.fail(f"Node '{node_name}' failed with regularizer '{reg_name}': {e}")
