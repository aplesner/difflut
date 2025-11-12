"""
Integration tests: Nodes with Initializers
Tests that each node type works with each registered initializer.
"""

import pytest
import torch
from testing_utils import (
    IgnoreWarnings,
    generate_uniform_input,
    instantiate_node,
)

from difflut.registry import REGISTRY


def _get_node_type_name(node_class):
    """Get the node type name for residual initialization."""
    name = node_class.__name__

    # Map specific node class names to their registered node_type strings
    node_type_map = {
        "DWNNode": "dwn",
        "DWNStableNode": "dwn_stable",
        "ProbabilisticNode": "probabilistic",
        "HybridNode": "hybrid",
        "LinearLUTNode": "linear_lut",
        "PolyLUTNode": "polylut",
        "NeuralLUTNode": "neurallut",
        "FourierNode": "fourier",
        "DiffLogicNode": "difflogic",
    }

    return node_type_map.get(name, name.lower().replace("node", ""))


def _get_input_dim_for_node(node_name: str) -> int:
    """Get appropriate input dimension for node type.

    DiffLogic has exponential complexity (2^(2^n)), so we limit it to 2 inputs.
    Other nodes can use larger dimensions.
    """
    if node_name == "difflogic":
        return 2  # 2^(2^2) = 16 functions (manageable)
    return 4  # Default for other nodes


@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
@pytest.mark.parametrize("init_name", REGISTRY.list_initializers())
def test_node_with_initializer(node_name, init_name):
    """Test that each node works with each registered initializer."""
    node_class = REGISTRY.get_node(node_name)
    init_fn = REGISTRY.get_initializer(init_name)
    input_dim = _get_input_dim_for_node(node_name)

    # Skip residual init test if it requires special parameters
    # (residual_init needs node-specific kwargs)
    if init_name == "residual":
        pytest.skip(
            f"Residual initializer requires node-specific parameters. "
            f"Tested separately in test_residual_initialization.py"
        )

    try:
        with IgnoreWarnings():
            # Create node with initializer
            node = instantiate_node(node_class, input_dim=input_dim, output_dim=2, init_fn=init_fn)

        # Test forward pass works
        input_tensor = generate_uniform_input((8, input_dim), seed=42)
        with torch.no_grad():
            output = node(input_tensor)

        # Verify output shape and validity
        assert output.shape == (8, 2), (
            f"Node '{node_name}' with initializer '{init_name}' "
            f"produced wrong shape: {output.shape}"
        )
        assert not torch.isnan(
            output
        ).any(), f"Node '{node_name}' with initializer '{init_name}' produced NaN outputs"
        assert not torch.isinf(
            output
        ).any(), f"Node '{node_name}' with initializer '{init_name}' produced Inf outputs"

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
        pytest.fail(f"Node '{node_name}' failed with initializer '{init_name}': {e}")


@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
def test_node_with_residual_initializer(node_name):
    """Test that each node works with residual initializer and appropriate init_kwargs."""
    node_class = REGISTRY.get_node(node_name)
    init_fn = REGISTRY.get_initializer("residual")
    input_dim = _get_input_dim_for_node(node_name)

    node_type = _get_node_type_name(node_class)

    # Build appropriate init_kwargs based on node type
    init_kwargs = {
        "node_type": node_type,
        "noise_factor": 0.0,
        "logit_clarity": 5.0,
    }

    # Add input_dim for LUT-based nodes (required for truth table sizing)
    if node_type in ["dwn", "dwn_stable", "probabilistic", "hybrid", "linear_lut", "difflogic"]:
        init_kwargs["input_dim"] = input_dim

    # Add node-specific parameters
    if node_type == "polylut":
        # PolyLUT needs special handling - it adds monomial_combinations itself
        init_kwargs["monomial_combinations"] = None
        init_kwargs["input_dim"] = 4
    elif node_type == "neurallut":
        init_kwargs["layer_idx"] = 0
        init_kwargs["num_layers"] = 1
        init_kwargs["param_name"] = "weight"
    elif node_type == "fourier":
        init_kwargs["num_frequencies"] = 4
        init_kwargs["param_name"] = "amplitudes"

    try:
        with IgnoreWarnings():
            # Create node with residual initializer
            node = instantiate_node(
                node_class,
                input_dim=input_dim,
                output_dim=1,
                init_fn=init_fn,
                init_kwargs=init_kwargs,
            )

        # Test forward pass works
        input_tensor = generate_uniform_input((8, input_dim), seed=42)
        with torch.no_grad():
            output = node(input_tensor)

        # Verify output shape and validity
        assert output.shape == (8, 1), (
            f"Node '{node_name}' with residual initializer " f"produced wrong shape: {output.shape}"
        )
        assert not torch.isnan(
            output
        ).any(), f"Node '{node_name}' with residual initializer produced NaN outputs"
        assert not torch.isinf(
            output
        ).any(), f"Node '{node_name}' with residual initializer produced Inf outputs"

        # Verify parameters were initialized properly
        for param_name, param in node.named_parameters():
            assert not torch.isnan(param).any(), (
                f"Node '{node_name}' with residual initializer "
                f"has NaN in parameter '{param_name}'"
            )
            assert not torch.isinf(param).any(), (
                f"Node '{node_name}' with residual initializer "
                f"has Inf in parameter '{param_name}'"
            )

    except Exception as e:
        pytest.fail(f"Node '{node_name}' failed with residual initializer: {e}")
