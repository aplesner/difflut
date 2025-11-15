"""
Tests for SimpleFeedForward model.

Tests specific to feedforward model behavior:
- Classification tasks
- Encoder fitting on flat inputs
- Forward pass dimensions
- Gradient computation
- CPU/GPU consistency
"""

import pytest
import torch
from testing_utils import (IgnoreWarnings, assert_gradients_exist,
                           generate_uniform_input)

from difflut.models import ModelConfig, SimpleFeedForward


@pytest.fixture
def feedforward_config():
    """Create a minimal config for testing."""
    return ModelConfig(
        model_type="feedforward",
        params={
            "layer_type": "random",
            "node_type": "probabilistic",
            "encoder_config": {"name": "thermometer", "num_bits": 4},
            "node_input_dim": 6,
            "layer_widths": [32],  # Reduced from [128, 64] to single smaller layer
            "num_classes": 10,
            "dataset": "test",
            "input_size": 50,  # Reduced from 100
        },
    )


class TestSimpleFeedForwardBasics:
    """Test basic functionality of SimpleFeedForward."""

    def test_model_instantiation(self, feedforward_config):
        """Test that model can be instantiated with config."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with IgnoreWarnings():
            model = SimpleFeedForward(feedforward_config).to(device)
            assert model is not None
            assert hasattr(model, "encoder")
            assert hasattr(model, "layers")
            assert hasattr(model, "output_layer")

    def test_encoder_fitting(self, feedforward_config):
        """Test that encoder fitting works."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with IgnoreWarnings():
            model = SimpleFeedForward(feedforward_config).to(device)
            data = generate_uniform_input((8, 50)).to(device)  # Reduced batch size

            assert not model.encoder_fitted
            model.fit_encoder(data)
            assert model.encoder_fitted

    def test_forward_pass_classification(self, feedforward_config):
        """Test forward pass produces correct output shape for classification."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with IgnoreWarnings():
            model = SimpleFeedForward(feedforward_config).to(device)
            data = generate_uniform_input((8, 50)).to(device)
            model.fit_encoder(data)

            # Forward pass
            batch = generate_uniform_input((4, 50), seed=42).to(device)
            with torch.no_grad():
                output = model(batch)

            # Check shape: (batch_size, num_classes)
            assert output.shape == (4, 10)
            assert isinstance(output, torch.Tensor)

    def test_gradients_computation(self, feedforward_config):
        """Test that gradients are computed correctly."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with IgnoreWarnings():
            model = SimpleFeedForward(feedforward_config).to(device)
            data = generate_uniform_input((8, 50)).to(device)
            model.fit_encoder(data)

            model.train()
            batch = generate_uniform_input((2, 50), seed=42).to(device)
            batch = batch.requires_grad_(True)

            output = model(batch)
            loss = output.sum()
            loss.backward()

            # Check gradients exist
            assert_gradients_exist(model)

    @pytest.mark.gpu
    def test_cpu_gpu_consistency(self, feedforward_config):
        """Test that CPU and GPU give same results."""
        with IgnoreWarnings():
            # Create models with same seed before moving to device
            torch.manual_seed(42)
            model_cpu = SimpleFeedForward(feedforward_config)
            model_cpu = model_cpu.cpu()

            torch.manual_seed(42)
            model_gpu = SimpleFeedForward(feedforward_config)
            model_gpu = model_gpu.cuda()

            # Fit encoders - reduced batch size
            data_cpu = generate_uniform_input((8, 50), seed=42).cpu()
            data_gpu = generate_uniform_input(
                (8, 50), seed=42
            ).cuda()  # Same seed for same data

            model_cpu.fit_encoder(data_cpu)
            model_gpu.fit_encoder(data_gpu)

            # Forward pass - reduced batch size
            model_cpu.eval()
            model_gpu.eval()

            input_cpu = generate_uniform_input((4, 50), seed=123).cpu()
            input_gpu = generate_uniform_input(
                (4, 50), seed=123
            ).cuda()  # Same seed for same data

            with torch.no_grad():
                output_cpu = model_cpu(input_cpu)
                output_gpu = model_gpu(input_gpu).cpu()

            # Compare outputs - allow for minor numerical differences between CPU and GPU
            torch.testing.assert_close(output_cpu, output_gpu, atol=1e-4, rtol=1e-3)


class TestSimpleFeedForwardInputSizes:
    """Test different input sizes."""

    @pytest.mark.parametrize("input_size", [50, 100, 784, 1024])
    def test_different_input_sizes(self, input_size, device):
        """Test model works with different input sizes."""
        config = ModelConfig(
            model_type="feedforward",
            params={
                "layer_type": "random",
                "node_type": "probabilistic",
                "encoder_config": {"name": "thermometer", "num_bits": 4},
                "node_input_dim": 6,
                "layer_widths": [32],  # Reduced
                "num_classes": 10,
                "input_size": input_size,
            },
        )

        with IgnoreWarnings():
            model = SimpleFeedForward(config).to(device)
            data = generate_uniform_input((4, input_size)).to(device)
            model.fit_encoder(data)

            batch = generate_uniform_input((2, input_size)).to(device)
            with torch.no_grad():
                output = model(batch)

            assert output.shape == (2, 10)


class TestSimpleFeedForwardNumClasses:
    """Test different number of output classes."""

    @pytest.mark.parametrize("num_classes", [2, 10, 100, 1000])
    def test_different_num_classes(self, num_classes, device):
        """Test model works with different number of classes."""
        config = ModelConfig(
            model_type="feedforward",
            params={
                "layer_type": "random",
                "node_type": "probabilistic",
                "encoder_config": {"name": "thermometer", "num_bits": 4},
                "node_input_dim": 6,
                "layer_widths": [32],  # Reduced
                "num_classes": num_classes,
                "input_size": 50,  # Reduced
            },
        )

        with IgnoreWarnings():
            model = SimpleFeedForward(config).to(device)
            data = generate_uniform_input((4, 50)).to(device)
            model.fit_encoder(data)

            batch = generate_uniform_input((2, 50)).to(device)
            with torch.no_grad():
                output = model(batch)

            assert output.shape == (2, num_classes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
