"""
Comprehensive forward pass tests for all layer types.
Tests: shape correctness, output range [0,1], CPU/GPU consistency, gradients.

Uses pytest parametrization for individual test discovery.
"""

import pytest
import torch
import torch.nn as nn
from difflut.registry import REGISTRY
from testing_utils import (
    is_cuda_available,
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


# ============================================================================
# Layer Forward Pass Tests
# ============================================================================

@pytest.mark.parametrize("layer_name", REGISTRY.list_layers())
class TestLayerForwardPass:
    """Comprehensive forward pass tests for layers."""
    
    def test_shape_correct(self, layer_name):
        """Test 2.1: Forward pass produces correct output shape."""
        layer_class = REGISTRY.get_layer(layer_name)
        
        with IgnoreWarnings():
            layer = instantiate_layer(
                layer_class,
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
    
    def test_output_range_01(self, layer_name):
        """Test 2.2: Output range is [0, 1]."""
        layer_class = REGISTRY.get_layer(layer_name)
        
        with IgnoreWarnings():
            layer = instantiate_layer(
                layer_class,
                input_size=256,
                output_size=128,
                n=4
            )
        layer.eval()
        
        # Test multiple random inputs
        for seed in [42, 123, 456]:
            input_tensor = generate_uniform_input((8, 256), seed=seed)
            with torch.no_grad():
                output = layer(input_tensor)
            
            assert_range(output, 0.0, 1.0)
    
    @pytest.mark.gpu
    def test_cpu_gpu_consistency(self, layer_name):
        """Test 2.3: CPU and GPU implementations give same forward pass."""
        if not is_cuda_available():
            pytest.skip("CUDA not available")
        
        layer_class = REGISTRY.get_layer(layer_name)
        
        with IgnoreWarnings():
            layer_cpu = instantiate_layer(
                layer_class,
                input_size=256,
                output_size=128,
                n=4
            )
        
        input_cpu = generate_uniform_input((8, 256), seed=42)
        
        try:
            output_cpu, output_gpu = compare_cpu_gpu_forward(
                layer_cpu, input_cpu, atol=CPU_GPU_ATOL, rtol=CPU_GPU_RTOL
            )
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                pytest.skip(f"CUDA error for {layer_name}: {e}")
            raise
    
    def test_gradients_exist(self, layer_name):
        """Test 2.4: Gradients exist and are not all zero."""
        layer_class = REGISTRY.get_layer(layer_name)
        
        with IgnoreWarnings():
            layer = instantiate_layer(
                layer_class,
                input_size=256,
                output_size=128,
                n=4
            )
        
        layer.train()
        input_tensor = generate_uniform_input((8, 256), seed=42)
        input_tensor.requires_grad = True
        
        output = layer(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist for parameters
        assert_gradients_exist(layer)


# ============================================================================
# Layer with Different Node Types
# ============================================================================

@pytest.mark.parametrize("layer_name", REGISTRY.list_layers())
@pytest.mark.parametrize("node_name", REGISTRY.list_nodes())
def test_layer_with_node_type(layer_name, node_name):
    """Test layer works with all node types."""
    layer_class = REGISTRY.get_layer(layer_name)
    node_class = REGISTRY.get_node(node_name)
    
    try:
        with IgnoreWarnings():
            layer = instantiate_layer(
                layer_class,
                input_size=256,
                output_size=128,
                node_type=node_class,
                n=4
            )
        
        # Test forward pass
        input_tensor = generate_uniform_input((8, 256), seed=42)
        with torch.no_grad():
            output = layer(input_tensor)
        
        assert output.shape == (8, 128), \
            f"{layer_name} with {node_name} produced wrong shape"
        
    except TypeError as e:
        # Some layer/node combinations might not be compatible
        if "node_type" in str(e):
            pytest.skip(f"{layer_name} does not support node_type parameter")
        raise


# ============================================================================
# Additional Layer Tests
# ============================================================================

@pytest.mark.parametrize("layer_name", REGISTRY.list_layers())
def test_layer_different_sizes(layer_name):
    """Test layer works with different input/output sizes."""
    layer_class = REGISTRY.get_layer(layer_name)
    
    test_configs = [
        (64, 32),    # Small
        (256, 128),  # Medium
        (512, 256),  # Large
    ]
    
    for input_size, output_size in test_configs:
        with IgnoreWarnings():
            layer = instantiate_layer(
                layer_class,
                input_size=input_size,
                output_size=output_size,
                n=4
            )
        
        input_tensor = generate_uniform_input((8, input_size), seed=42)
        with torch.no_grad():
            output = layer(input_tensor)
        
        assert output.shape == (8, output_size), \
            f"{layer_name} failed for sizes ({input_size}, {output_size})"


@pytest.mark.parametrize("layer_name", REGISTRY.list_layers())
def test_layer_different_batch_sizes(layer_name):
    """Test layer works with different batch sizes."""
    layer_class = REGISTRY.get_layer(layer_name)
    
    with IgnoreWarnings():
        layer = instantiate_layer(
            layer_class,
            input_size=256,
            output_size=128,
            n=4
        )
    layer.eval()
    
    for batch_size in [1, 8, 32, 128]:
        input_tensor = generate_uniform_input((batch_size, 256), seed=42)
        with torch.no_grad():
            output = layer(input_tensor)
        
        assert output.shape == (batch_size, 128), \
            f"{layer_name} failed for batch_size={batch_size}"
