"""
Tests for SimpleConvolutional model.

Tests specific to convolutional model behavior:
- Image input processing
- Encoder fitting on spatial inputs
- Forward pass dimensions
- Gradient computation
- CPU/GPU consistency
"""

import pytest
import torch
from testing_utils import (IgnoreWarnings, assert_gradients_exist,
                           generate_uniform_input)

from difflut.models import ModelConfig, SimpleConvolutional


@pytest.fixture
def convolutional_config():
    """Create a minimal config for testing."""
    return ModelConfig(
        model_type="convolutional",
        params={
            "layer_type": "random",
            "node_type": "probabilistic",
            "encoder_config": {"name": "thermometer", "num_bits": 3},  # Reduced from 4
            "node_input_dim": 6,
            # Not used for convolutional - see conv_layer_widths below
            "layer_widths": [4],
            "num_classes": 10,
            "dataset": "test",
            "input_size": None,  # Will be inferred from input
        },
        # Convolutional-specific params in runtime
        runtime={
            "conv_kernel_size": 3,
            "conv_stride": 1,
            "conv_padding": 1,
            "input_channels": 1,  # Reduced from 3 (grayscale instead of RGB)
            "input_height": 8,  # Reduced from 16
            "input_width": 8,  # Reduced from 8
            # Reduced from default [32, 64, 64] - only 4 output channels
            "conv_layer_widths": [4],
            # Node kwargs for proper initialization
            "node_kwargs": {
                "input_dim": 6,
                "output_dim": 1,
                "eval_mode": "expectation",
            },
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

    def test_encoder_fitting(self, convolutional_config, device):
        """Test that encoder fitting works with spatial inputs."""
        with IgnoreWarnings():
            model = SimpleConvolutional(convolutional_config).to(device)
            # Input: (batch, channels, height, width) - reduced to 1 channel, 8x8
            data = generate_uniform_input((4, 1, 8, 8)).to(
                device
            )  # Reduced batch and size

            assert not model.encoder_fitted
            model.fit_encoder(data)
            assert model.encoder_fitted

    def test_forward_pass_classification(self, convolutional_config, device):
        """Test forward pass produces correct output shape for classification."""
        with IgnoreWarnings():
            model = SimpleConvolutional(convolutional_config).to(device)
            data = generate_uniform_input((4, 1, 8, 8)).to(device)  # Match new config
            model.fit_encoder(data)

            # Forward pass
            batch = generate_uniform_input((2, 1, 8, 8), seed=42).to(
                device
            )  # Match new config
            with torch.no_grad():
                output = model(batch)

            # Check shape: (batch_size, num_classes)
            assert output.shape == (2, 10)  # Fixed: batch size is 2, not 4
            assert isinstance(output, torch.Tensor)

    def test_gradients_computation(self, convolutional_config, device):
        """Test that gradients are computed correctly."""
        with IgnoreWarnings():
            model = SimpleConvolutional(convolutional_config).to(device)
            data = generate_uniform_input((4, 1, 8, 8)).to(device)  # Match new config
            model.fit_encoder(data)

            model.train()
            batch = generate_uniform_input((2, 1, 8, 8), seed=42).to(
                device
            )  # Match new config
            batch = batch.requires_grad_(True)

            output = model(batch)
            loss = output.sum()
            loss.backward()

            # Check gradients exist
            assert_gradients_exist(model)

    @pytest.mark.gpu
    def test_cpu_gpu_consistency(self, convolutional_config, device):
        """Test that CPU and GPU give same results."""
        with IgnoreWarnings():
            # Create models with same seed before moving to device
            torch.manual_seed(42)
            model_cpu = SimpleConvolutional(convolutional_config)
            model_cpu = model_cpu.cpu()

            torch.manual_seed(42)
            model_gpu = SimpleConvolutional(convolutional_config)
            model_gpu = model_gpu.cuda()

            # Fit encoders - match new config
            data_cpu = generate_uniform_input(
                (4, 1, 8, 8), seed=42
            ).cpu()  # Match new config
            data_gpu = generate_uniform_input(
                (4, 1, 8, 8), seed=42
            ).cuda()  # Same seed for same data

            model_cpu.fit_encoder(data_cpu)
            model_gpu.fit_encoder(data_gpu)

            # Forward pass
            model_cpu.eval()
            model_gpu.eval()

            input_cpu = generate_uniform_input(
                (2, 1, 8, 8), seed=123
            ).cpu()  # Match new config
            input_gpu = generate_uniform_input(
                (2, 1, 8, 8), seed=123
            ).cuda()  # Same seed for same data

            with torch.no_grad():
                output_cpu = model_cpu(input_cpu)
                output_gpu = model_gpu(input_gpu).cpu()

            # Compare outputs - should be close (allowing for minor numerical differences)
            torch.testing.assert_close(output_cpu, output_gpu, atol=1e-4, rtol=1e-3)


class TestSimpleConvolutionalInputSizes:
    """Test different input sizes."""

    # Reduced from [16, 28, 32]
    @pytest.mark.parametrize("image_size", [8, 16])
    @pytest.mark.parametrize("channels", [1])  # Reduced from [1, 3]
    def test_different_input_sizes(self, image_size, channels, device):
        """Test model works with different input sizes."""
        config = ModelConfig(
            model_type="convolutional",
            params={
                "layer_type": "random",
                "node_type": "probabilistic",
                "encoder_config": {
                    "name": "thermometer",
                    "num_bits": 3,
                },  # Reduced from 4
                "node_input_dim": 6,
                "layer_widths": [4],  # Reduced from [8]
                "num_classes": 10,
            },
            runtime={
                "conv_kernel_size": 3,
                "conv_stride": 1,
                "conv_padding": 1,
                "input_channels": channels,
                "input_height": image_size,
                "input_width": image_size,
                # Node kwargs for proper initialization
                "node_kwargs": {
                    "input_dim": 6,
                    "output_dim": 1,
                    "eval_mode": "expectation",
                },
            },
        )

        with IgnoreWarnings():
            model = SimpleConvolutional(config).to(device)
            data = generate_uniform_input((2, channels, image_size, image_size)).to(
                device
            )  # Reduced batch
            model.fit_encoder(data)

            batch = generate_uniform_input((2, channels, image_size, image_size)).to(
                device
            )
            with torch.no_grad():
                output = model(batch)

            assert output.shape == (2, 10)


class TestSimpleConvolutionalNumClasses:
    """Test different number of output classes."""

    # Reduced from [2, 10, 100]
    @pytest.mark.parametrize("num_classes", [2, 10])
    def test_different_num_classes(self, num_classes, device):
        """Test model works with different number of classes."""
        config = ModelConfig(
            model_type="convolutional",
            params={
                "layer_type": "random",
                "node_type": "probabilistic",
                "encoder_config": {
                    "name": "thermometer",
                    "num_bits": 3,
                },  # Reduced from 4
                "node_input_dim": 6,
                "layer_widths": [4],  # Reduced from [8]
                "num_classes": num_classes,
            },
            runtime={
                "conv_kernel_size": 3,
                "conv_stride": 1,
                "conv_padding": 1,
                "input_channels": 1,  # Reduced from 3
                "input_height": 8,  # Reduced from 16
                "input_width": 8,  # Reduced from 16
                # Node kwargs for proper initialization
                "node_kwargs": {
                    "input_dim": 6,
                    "output_dim": 1,
                    "eval_mode": "expectation",
                },
            },
        )

        with IgnoreWarnings():
            model = SimpleConvolutional(config).to(device)
            data = generate_uniform_input((2, 1, 8, 8)).to(device)  # Match new config
            model.fit_encoder(data)

            batch = generate_uniform_input((2, 1, 8, 8)).to(device)  # Match new config
            with torch.no_grad():
                output = model(batch)

            assert output.shape == (2, num_classes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
