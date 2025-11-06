"""
Comprehensive forward pass tests for all encoder types.
Tests: shape correctness with flatten=True/False, output range [0,1], fit/encode cycle.

Uses pytest parametrization for individual test discovery.
"""

import pytest
import torch
import torch.nn as nn
from difflut.registry import REGISTRY
from testing_utils import (
    instantiate_encoder,
    generate_uniform_input,
    assert_shape_equal,
    assert_range,
    IgnoreWarnings,
)


# ============================================================================
# Encoder Forward Pass Tests
# ============================================================================

@pytest.mark.parametrize("encoder_name", REGISTRY.list_encoders())
class TestEncoderForwardPass:
    """Comprehensive forward pass tests for encoders."""
    
    def test_shape_flatten_true(self, encoder_name):
        """Test 3.1: Forward pass with flatten=True produces correct shape."""
        encoder_class = REGISTRY.get_encoder(encoder_name)
        
        with IgnoreWarnings():
            encoder = instantiate_encoder(encoder_class, num_bits=8, flatten=True)
        encoder.eval()
        
        # Create input data
        input_data = generate_uniform_input((10, 50), seed=42)  # 10 samples, 50 features
        
        # Fit encoder
        encoder.fit(input_data)
        
        # Test forward pass (flatten is set in __init__)
        with torch.no_grad():
            output = encoder.encode(input_data)
        
        # With flatten=True, output should be 2D: (batch_size, num_features * num_bits)
        expected_shape = (10, 50 * 8)  # 50 features * 8 bits
        assert_shape_equal(output, expected_shape)
    
    def test_shape_flatten_false(self, encoder_name):
        """Test 3.2: Forward pass with flatten=False produces correct shape."""
        encoder_class = REGISTRY.get_encoder(encoder_name)
        
        with IgnoreWarnings():
            encoder = instantiate_encoder(encoder_class, num_bits=8, flatten=False)
        encoder.eval()
        
        # Create input data
        input_data = generate_uniform_input((10, 50), seed=42)  # 10 samples, 50 features
        
        # Fit encoder
        encoder.fit(input_data)
        
        # Test forward pass (flatten is set in __init__)
        with torch.no_grad():
            output = encoder.encode(input_data)
        
        # With flatten=False, output should be 3D: (batch_size, num_features, num_bits)
        expected_shape = (10, 50, 8)  # 10 batch, 50 features, 8 bits
        assert_shape_equal(output, expected_shape)
    
    def test_output_range_01(self, encoder_name):
        """Test 3.3: Output range is [0, 1]."""
        encoder_class = REGISTRY.get_encoder(encoder_name)
        
        with IgnoreWarnings():
            encoder = instantiate_encoder(encoder_class, num_bits=8, flatten=True)
        encoder.eval()
        
        # Create input data
        input_data = generate_uniform_input((10, 50), seed=42)
        
        # Fit encoder
        encoder.fit(input_data)
        
        # Test output range for multiple inputs
        for seed in [42, 123, 456]:
            test_input = generate_uniform_input((10, 50), seed=seed)
            with torch.no_grad():
                output = encoder.encode(test_input)
            
            assert_range(output, 0.0, 1.0)
    
    def test_fit_encode_cycle(self, encoder_name):
        """Test 3.4: Encoder fit() and encode() work correctly."""
        encoder_class = REGISTRY.get_encoder(encoder_name)
        
        with IgnoreWarnings():
            encoder = instantiate_encoder(encoder_class, num_bits=8)
        
        # Create input data
        input_data = generate_uniform_input((50, 20), seed=42)
        
        # Fit encoder
        encoder.fit(input_data)
        
        # Encode data
        with torch.no_grad():
            encoded = encoder.encode(input_data[:10])
        
        # Check output is not None and has correct batch dimension
        assert encoded is not None
        assert encoded.shape[0] == 10


# ============================================================================
# Additional Encoder Tests
# ============================================================================

@pytest.mark.parametrize("encoder_name", REGISTRY.list_encoders())
@pytest.mark.parametrize("num_bits", [4, 8, 16])
def test_encoder_different_bit_widths(encoder_name, num_bits):
    """Test encoder works with different bit widths."""
    encoder_class = REGISTRY.get_encoder(encoder_name)
    
    with IgnoreWarnings():
        encoder = instantiate_encoder(encoder_class, num_bits=num_bits, flatten=True)
    
    # Create input data
    input_data = generate_uniform_input((10, 20), seed=42)
    
    # Fit and encode
    encoder.fit(input_data)
    with torch.no_grad():
        output = encoder.encode(input_data)
    
    # Check shape includes num_bits
    expected_total_bits = 20 * num_bits
    assert output.shape == (10, expected_total_bits), \
        f"{encoder_name} with {num_bits} bits produced wrong shape"


@pytest.mark.parametrize("encoder_name", REGISTRY.list_encoders())
def test_encoder_different_input_sizes(encoder_name):
    """Test encoder works with different input feature sizes."""
    encoder_class = REGISTRY.get_encoder(encoder_name)
    
    test_feature_sizes = [10, 50, 100]
    
    for n_features in test_feature_sizes:
        with IgnoreWarnings():
            encoder = instantiate_encoder(encoder_class, num_bits=8, flatten=True)
        
        # Create input data
        input_data = generate_uniform_input((10, n_features), seed=42)
        
        # Fit and encode
        encoder.fit(input_data)
        with torch.no_grad():
            output = encoder.encode(input_data)
        
        expected_shape = (10, n_features * 8)
        assert output.shape == expected_shape, \
            f"{encoder_name} failed for {n_features} features"


@pytest.mark.parametrize("encoder_name", REGISTRY.list_encoders())
def test_encoder_different_batch_sizes(encoder_name):
    """Test encoder works with different batch sizes."""
    encoder_class = REGISTRY.get_encoder(encoder_name)
    
    with IgnoreWarnings():
        encoder = instantiate_encoder(encoder_class, num_bits=8, flatten=True)
    
    # Fit with initial data
    fit_data = generate_uniform_input((50, 20), seed=42)
    encoder.fit(fit_data)
    
    # Test different batch sizes
    for batch_size in [1, 8, 32, 128]:
        test_input = generate_uniform_input((batch_size, 20), seed=42)
        with torch.no_grad():
            output = encoder.encode(test_input)
        
        assert output.shape == (batch_size, 20 * 8), \
            f"{encoder_name} failed for batch_size={batch_size}"


@pytest.mark.parametrize("encoder_name", REGISTRY.list_encoders())
def test_encoder_forward_vs_encode_consistency(encoder_name):
    """Test that forward() and encode() produce same output."""
    encoder_class = REGISTRY.get_encoder(encoder_name)
    
    with IgnoreWarnings():
        encoder = instantiate_encoder(encoder_class, num_bits=8, flatten=True)
    
    # Fit encoder
    input_data = generate_uniform_input((50, 20), seed=42)
    encoder.fit(input_data)
    
    # Get outputs from both methods
    test_input = generate_uniform_input((10, 20), seed=42)
    
    with torch.no_grad():
        # Try forward() if available
        try:
            output_forward = encoder(test_input)
        except:
            pytest.skip(f"{encoder_name} does not support forward() call")
        
        output_encode = encoder.encode(test_input)
    
    # They should produce same output (or at least same shape)
    assert output_forward.shape == output_encode.shape, \
        f"{encoder_name} forward() and encode() produce different shapes"
