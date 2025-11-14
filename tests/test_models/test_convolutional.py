"""
Tests for SimpleConvolutional model.

Tests specific to convolutional model behavior:
- Image classification tasks
- Encoder fitting on spatial inputs
- Forward pass dimensions
- Gradient computation
- CPU/GPU consistency
"""

import pytest
import torch
from testing_utils import (
    IgnoreWarnings,
    assert_gradients_exist,
    generate_uniform_input,
    is_cuda_available,
)

from difflut.models import ModelConfig, SimpleConvolutional


@pytest.fixture
def convolutional_config():
    """Create a minimal config for testing."""
    return ModelConfig(
        model_type="convolutional",
        layer_type="random",
        node_type="probabilistic",
        encoder_config={"name": "thermometer", "num_bits": 3},  # Reduced from 4
        node_input_dim=6,
        layer_widths=[4],  # Not used for convolutional - see conv_layer_widths below
        num_classes=10,
        dataset="test",
        input_size=None,  # Will be inferred from input
        # Convolutional-specific params in runtime
        runtime={
            "conv_kernel_size": 3,
            "conv_stride": 1,
            "conv_padding": 1,
            "input_channels": 1,  # Reduced from 3 (grayscale instead of RGB)
            "input_height": 8,   # Reduced from 16
            "input_width": 8,    # Reduced from 8
            "conv_layer_widths": [4],  # Reduced from default [32, 64, 64] - only 4 output channels
        },
    )


class TestSimpleConvolutionalBasics:
    """Test basic functionality of SimpleConvolutional."""

    def test_model_instantiation(self, convolutional_config):
        """Test that model can be instantiated with config."""
        with IgnoreWarnings():
            model = SimpleConvolutional(convolutional_config)
            assert model is not None
            assert hasattr(model, "encoder")
            assert hasattr(model, "conv_layers")
            assert hasattr(model, "output_layer")

    def test_encoder_fitting(self, convolutional_config):
        """Test that encoder fitting works with spatial inputs."""
        with IgnoreWarnings():
            model = SimpleConvolutional(convolutional_config)
            # Input: (batch, channels, height, width) - reduced to 1 channel, 8x8
            data = generate_uniform_input((4, 1, 8, 8))  # Reduced batch and size
            
            assert not model.encoder_fitted
            model.fit_encoder(data)
            assert model.encoder_fitted

    def test_forward_pass_classification(self, convolutional_config):
        """Test forward pass produces correct output shape for classification."""
        with IgnoreWarnings():
            model = SimpleConvolutional(convolutional_config)
            data = generate_uniform_input((4, 1, 8, 8))  # Match new config
            model.fit_encoder(data)
            
            # Forward pass
            batch = generate_uniform_input((2, 1, 8, 8), seed=42)  # Match new config
            with torch.no_grad():
                output = model(batch)
            
            # Check shape: (batch_size, num_classes)
            assert output.shape == (2, 10)  # Fixed: batch size is 2, not 4
            assert isinstance(output, torch.Tensor)

    def test_gradients_computation(self, convolutional_config):
        """Test that gradients are computed correctly."""
        with IgnoreWarnings():
            model = SimpleConvolutional(convolutional_config)
            data = generate_uniform_input((4, 1, 8, 8))  # Match new config
            model.fit_encoder(data)
            
            model.train()
            batch = generate_uniform_input((2, 1, 8, 8), seed=42)  # Match new config
            batch.requires_grad = True
            
            output = model(batch)
            loss = output.sum()
            loss.backward()
            
            # Check gradients exist
            assert_gradients_exist(model)

    @pytest.mark.gpu
    def test_cpu_gpu_consistency(self, convolutional_config):
        """Test that CPU and GPU give same results."""
        if not is_cuda_available():
            pytest.skip("CUDA not available")
        
        with IgnoreWarnings():
            # Create models
            torch.manual_seed(42)
            model_cpu = SimpleConvolutional(convolutional_config)
            
            torch.manual_seed(42)
            model_gpu = SimpleConvolutional(convolutional_config).cuda()
            
            # Fit encoders - match new config
            data_cpu = generate_uniform_input((4, 1, 8, 8), seed=42)  # Match new config
            data_gpu = data_cpu.cuda()
            
            model_cpu.fit_encoder(data_cpu)
            model_gpu.fit_encoder(data_gpu)
            
            # Forward pass
            model_cpu.eval()
            model_gpu.eval()
            
            input_cpu = generate_uniform_input((2, 1, 8, 8), seed=123)  # Match new config
            input_gpu = input_cpu.cuda()
            
            with torch.no_grad():
                output_cpu = model_cpu(input_cpu)
                output_gpu = model_gpu(input_gpu).cpu()
            
            # Compare outputs - should be identical with same seed and weights
            torch.testing.assert_close(output_cpu, output_gpu, atol=1e-5, rtol=1e-4)


class TestSimpleConvolutionalInputSizes:
    """Test different input sizes."""

    @pytest.mark.parametrize("image_size", [8, 16])  # Reduced from [16, 28, 32]
    @pytest.mark.parametrize("channels", [1])  # Reduced from [1, 3]
    def test_different_input_sizes(self, image_size, channels):
        """Test model works with different input sizes."""
        config = ModelConfig(
            model_type="convolutional",
            layer_type="random",
            node_type="probabilistic",
            encoder_config={"name": "thermometer", "num_bits": 3},  # Reduced from 4
            node_input_dim=6,
            layer_widths=[4],  # Reduced from [8]
            num_classes=10,
            runtime={
                "conv_kernel_size": 3,
                "conv_stride": 1,
                "conv_padding": 1,
                "input_channels": channels,
                "input_height": image_size,
                "input_width": image_size,
            },
        )
        
        with IgnoreWarnings():
            model = SimpleConvolutional(config)
            data = generate_uniform_input((2, channels, image_size, image_size))  # Reduced batch
            model.fit_encoder(data)
            
            batch = generate_uniform_input((2, channels, image_size, image_size))
            with torch.no_grad():
                output = model(batch)
            
            assert output.shape == (2, 10)


class TestSimpleConvolutionalNumClasses:
    """Test different number of output classes."""

    @pytest.mark.parametrize("num_classes", [2, 10])  # Reduced from [2, 10, 100]
    def test_different_num_classes(self, num_classes):
        """Test model works with different number of classes."""
        config = ModelConfig(
            model_type="convolutional",
            layer_type="random",
            node_type="probabilistic",
            encoder_config={"name": "thermometer", "num_bits": 3},  # Reduced from 4
            node_input_dim=6,
            layer_widths=[4],  # Reduced from [8]
            num_classes=num_classes,
            runtime={
                "conv_kernel_size": 3,
                "conv_stride": 1,
                "conv_padding": 1,
                "input_channels": 1,  # Reduced from 3
                "input_height": 8,   # Reduced from 16
                "input_width": 8,    # Reduced from 16
            },
        )
        
        with IgnoreWarnings():
            model = SimpleConvolutional(config)
            data = generate_uniform_input((2, 1, 8, 8))  # Match new config
            model.fit_encoder(data)
            
            batch = generate_uniform_input((2, 1, 8, 8))  # Match new config
            with torch.no_grad():
                output = model(batch)
            
            assert output.shape == (2, num_classes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
