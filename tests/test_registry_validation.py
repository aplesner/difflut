"""
Registry validation tests.
Ensures all registered components are actually implemented and can be instantiated.

Uses pytest parametrization for individual test discovery.
"""

import pytest
import torch.nn as nn
from difflut.registry import REGISTRY
from testing_utils import (
    instantiate_node,
    instantiate_layer,
    instantiate_encoder,
    IgnoreWarnings
)


# ============================================================================
# Node Tests
# ============================================================================

@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
def test_node_is_implemented(node_name):
    """Test that registered node can be instantiated."""
    node_class = REGISTRY.get_node(node_name)
    assert node_class is not None, f"Node {node_name} class is None"
    
    with IgnoreWarnings():
        node = instantiate_node(node_class, input_dim=4, output_dim=1, layer_size=2)
    
    assert isinstance(node, nn.Module), f"Node {node_name} is not an nn.Module"
    assert hasattr(node, 'forward'), f"Node {node_name} has no forward method"


# ============================================================================
# Layer Tests
# ============================================================================

@pytest.mark.parametrize("layer_name", REGISTRY.list_layers())
def test_layer_is_implemented(layer_name):
    """Test that registered layer can be instantiated."""
    layer_class = REGISTRY.get_layer(layer_name)
    assert layer_class is not None, f"Layer {layer_name} class is None"

    with IgnoreWarnings():
        layer = instantiate_layer(layer_class, input_size=256, output_size=128, n=4)

    assert isinstance(layer, nn.Module), f"Layer {layer_name} is not an nn.Module"
    assert hasattr(layer, 'forward'), f"Layer {layer_name} has no forward method"


# ============================================================================
# Encoder Tests
# ============================================================================

@pytest.mark.parametrize("encoder_name", REGISTRY.list_encoders())
def test_encoder_is_implemented(encoder_name):
    """Test that registered encoder can be instantiated."""
    encoder_class = REGISTRY.get_encoder(encoder_name)
    assert encoder_class is not None, f"Encoder {encoder_name} class is None"
    
    with IgnoreWarnings():
        encoder = instantiate_encoder(encoder_class, num_bits=8)
    
    assert isinstance(encoder, nn.Module), f"Encoder {encoder_name} is not an nn.Module"
    assert hasattr(encoder, 'forward'), f"Encoder {encoder_name} has no forward method"


# ============================================================================
# Registry Consistency Tests
# ============================================================================

def test_registry_list_all_structure():
    """Test that list_all() returns expected structure."""
    all_components = REGISTRY.list_all()
    expected_keys = {'nodes', 'layers', 'convolutional_layers', 'encoders', 'initializers', 'regularizers'}
    assert set(all_components.keys()) == expected_keys, \
        f"list_all() has keys {set(all_components.keys())}, expected {expected_keys}"


def test_registry_list_nodes_consistency():
    """Test that list_nodes() is consistent with list_all()."""
    nodes_direct = REGISTRY.list_nodes()
    nodes_from_all = REGISTRY.list_all()['nodes']
    assert nodes_direct == nodes_from_all, \
        "list_nodes() inconsistent with list_all()['nodes']"


def test_registry_list_layers_consistency():
    """Test that list_layers() is consistent with list_all()."""
    layers_direct = REGISTRY.list_layers()
    layers_from_all = REGISTRY.list_all()['layers']
    assert layers_direct == layers_from_all, \
        "list_layers() inconsistent with list_all()['layers']"


def test_registry_list_encoders_consistency():
    """Test that list_encoders() is consistent with list_all()."""
    encoders_direct = REGISTRY.list_encoders()
    encoders_from_all = REGISTRY.list_all()['encoders']
    assert encoders_direct == encoders_from_all, \
        "list_encoders() inconsistent with list_all()['encoders']"


def test_registry_get_node_works_for_all():
    """Test that get_node() works for all listed nodes."""
    for node_name in REGISTRY.list_nodes():
        node_class = REGISTRY.get_node(node_name)
        assert node_class is not None, f"get_node('{node_name}') returned None"


def test_registry_get_layer_works_for_all():
    """Test that get_layer() works for all listed layers."""
    for layer_name in REGISTRY.list_layers():
        layer_class = REGISTRY.get_layer(layer_name)
        assert layer_class is not None, f"get_layer('{layer_name}') returned None"


def test_registry_get_convolutional_layer_works_for_all():
    """Test that get_convolutional_layer() works for all listed convolutional layers."""
    for conv_layer_name in REGISTRY.list_convolutional_layers():
        conv_layer_class = REGISTRY.get_convolutional_layer(conv_layer_name)
        assert conv_layer_class is not None, f"get_convolutional_layer('{conv_layer_name}') returned None"


def test_registry_get_encoder_works_for_all():
    """Test that get_encoder() works for all listed encoders."""
    for encoder_name in REGISTRY.list_encoders():
        encoder_class = REGISTRY.get_encoder(encoder_name)
        assert encoder_class is not None, f"get_encoder('{encoder_name}') returned None"


def test_registry_invalid_node_raises_error():
    """Test that getting invalid node raises ValueError."""
    with pytest.raises(ValueError):
        REGISTRY.get_node('nonexistent_node_12345')


def test_registry_invalid_layer_raises_error():
    """Test that getting invalid layer raises ValueError."""
    with pytest.raises(ValueError):
        REGISTRY.get_layer('nonexistent_layer_12345')


def test_registry_invalid_encoder_raises_error():
    """Test that getting invalid encoder raises ValueError."""
    with pytest.raises(ValueError):
        REGISTRY.get_encoder('nonexistent_encoder_12345')

def test_registry_invalid_convolutional_layer_raises_error():
    """Test that getting invalid convolutional layer raises ValueError."""
    with pytest.raises(ValueError):
        REGISTRY.get_convolutional_layer('nonexistent_convolutional_layer_12345')
