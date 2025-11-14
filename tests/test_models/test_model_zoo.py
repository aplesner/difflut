"""
Comprehensive tests for DiffLUT Model Zoo.

Tests cover:
- Model instantiation
- Encoder fitting
- Forward pass dimension validation
- Regularization loss computation
- Parameter counting
- Model serialization
"""

import tempfile
from pathlib import Path

import pytest
import torch

from difflut import REGISTRY
from difflut.models import (
    BaseModel,
    CIFAR10Conv,
    MNISTDWNSmall,
    MNISTLinearSmall,
    get_model,
    list_models,
)


class TestModelInstantiation:
    """Test that all models can be instantiated."""

    def test_list_models(self):
        """Test that we can list all available models."""
        models = list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "mnist_fc_8k_linear" in models
        assert "cifar10_conv" in models

    def test_get_model_by_name(self):
        """Test getting models by name."""
        model = get_model("mnist_fc_8k_linear")
        assert isinstance(model, BaseModel)
        assert model.model_name == "mnist_fc_8k_linear"

    def test_get_model_invalid_name(self):
        """Test error handling for invalid model names."""
        with pytest.raises(ValueError):
            get_model("nonexistent_model")

    def test_mnist_linear_direct(self):
        """Test direct instantiation of MNIST linear model."""
        model = MNISTLinearSmall()
        assert isinstance(model, BaseModel)
        assert model.num_classes == 10
        assert model.input_size == 784

    def test_mnist_dwn_direct(self):
        """Test direct instantiation of MNIST DWN model."""
        model = MNISTDWNSmall(use_cuda=False)
        assert isinstance(model, BaseModel)
        assert model.num_classes == 10

    def test_cifar10_conv_direct(self):
        """Test direct instantiation of CIFAR-10 convolutional model."""
        model = CIFAR10Conv(use_cuda=False)
        assert isinstance(model, BaseModel)
        assert model.num_classes == 10
        assert model.input_size == 3072


class TestEncoderFitting:
    """Test encoder fitting workflow."""

    @pytest.fixture
    def mnist_data(self):
        """Generate random MNIST-like data."""
        return torch.randn(50, 28, 28)  # 50 random images

    @pytest.fixture
    def cifar_data(self):
        """Generate random CIFAR-10-like data."""
        return torch.randn(32, 3, 32, 32)  # 32 random RGB images

    def test_fit_encoder_mnist(self, mnist_data):
        """Test fitting encoder on MNIST data."""
        model = MNISTLinearSmall()
        assert not model.encoder_fitted

        model.fit_encoder(mnist_data)

        assert model.encoder_fitted
        assert hasattr(model, "encoded_input_size")
        assert model.encoded_input_size > 0

    def test_fit_encoder_cifar(self, cifar_data):
        """Test fitting encoder on CIFAR-10 data."""
        model = CIFAR10Conv(use_cuda=False)
        assert not model.encoder_fitted

        model.fit_encoder(cifar_data)

        assert model.encoder_fitted
        assert hasattr(model, "encoded_input_size")
        assert model.encoded_input_size > 0

    def test_fit_encoder_twice_raises_error(self, mnist_data):
        """Test that fitting encoder twice raises error."""
        model = MNISTLinearSmall()
        model.fit_encoder(mnist_data)

        with pytest.raises(RuntimeError):
            model.fit_encoder(mnist_data)

    def test_forward_without_fitting_raises_error(self, mnist_data):
        """Test that forward pass before fitting raises error."""
        model = MNISTLinearSmall()

        with pytest.raises(RuntimeError):
            _ = model(mnist_data)


class TestForwardPass:
    """Test forward pass dimension validation."""

    @pytest.fixture
    def mnist_model_fitted(self):
        """Create fitted MNIST model."""
        model = MNISTLinearSmall()
        data = torch.randn(32, 28, 28)
        model.fit_encoder(data)
        return model

    @pytest.fixture
    def cifar_model_fitted(self):
        """Create fitted CIFAR model."""
        model = CIFAR10Conv(use_cuda=False)
        data = torch.randn(16, 3, 32, 32)
        model.fit_encoder(data)
        return model

    def test_mnist_forward_shape(self, mnist_model_fitted):
        """Test MNIST forward pass output shape."""
        batch = torch.randn(16, 28, 28)
        output = mnist_model_fitted(batch)

        assert output.shape == (16, 10)  # Batch size, num classes

    def test_mnist_forward_flattened(self, mnist_model_fitted):
        """Test MNIST forward pass with flattened input."""
        batch = torch.randn(16, 784)
        output = mnist_model_fitted(batch)

        assert output.shape == (16, 10)

    def test_cifar_forward_shape(self, cifar_model_fitted):
        """Test CIFAR forward pass output shape."""
        batch = torch.randn(8, 3, 32, 32)
        output = cifar_model_fitted(batch)

        assert output.shape == (8, 10)

    def test_forward_batch_size_one(self, mnist_model_fitted):
        """Test forward pass with batch size 1."""
        batch = torch.randn(1, 28, 28)
        output = mnist_model_fitted(batch)

        assert output.shape == (1, 10)

    def test_forward_large_batch(self, mnist_model_fitted):
        """Test forward pass with large batch."""
        batch = torch.randn(256, 28, 28)
        output = mnist_model_fitted(batch)

        assert output.shape == (256, 10)


class TestRegularization:
    """Test regularization loss computation."""

    @pytest.fixture
    def model_with_bitflip(self):
        """Create model with bit flipping."""
        from difflut.models.comparison import MNISTBitFlip10

        model = MNISTBitFlip10()
        data = torch.randn(32, 28, 28)
        model.fit_encoder(data)
        return model

    def test_get_regularization_loss_zero(self):
        """Test that models without regularizers return zero loss."""
        model = MNISTLinearSmall()
        data = torch.randn(32, 28, 28)
        model.fit_encoder(data)

        reg_loss = model.get_regularization_loss()

        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.shape == torch.Size([])  # Scalar tensor

    def test_regularization_loss_is_scalar(self, model_with_bitflip):
        """Test that regularization loss is a scalar."""
        reg_loss = model_with_bitflip.get_regularization_loss()

        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.numel() == 1  # Single value


class TestParameterCounting:
    """Test parameter counting functionality."""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model."""
        model = MNISTLinearSmall()
        data = torch.randn(32, 28, 28)
        model.fit_encoder(data)
        return model

    def test_count_parameters_returns_dict(self, fitted_model):
        """Test that count_parameters returns dict."""
        counts = fitted_model.count_parameters()

        assert isinstance(counts, dict)
        assert "total" in counts
        assert "trainable" in counts
        assert "non_trainable" in counts

    def test_count_parameters_values(self, fitted_model):
        """Test that parameter counts are reasonable."""
        counts = fitted_model.count_parameters()

        assert counts["total"] > 0
        assert counts["trainable"] > 0
        assert counts["non_trainable"] >= 0
        assert counts["total"] == counts["trainable"] + counts["non_trainable"]

    def test_count_parameters_all_trainable(self, fitted_model):
        """Test that all parameters are trainable by default."""
        counts = fitted_model.count_parameters()

        # For untrained models, all should be trainable
        assert counts["trainable"] == counts["total"]
        assert counts["non_trainable"] == 0


class TestLayerTopology:
    """Test layer topology tracking."""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model."""
        model = MNISTLinearSmall()
        data = torch.randn(32, 28, 28)
        model.fit_encoder(data)
        return model

    def test_get_layer_topology(self, fitted_model):
        """Test getting layer topology."""
        topology = fitted_model.get_layer_topology()

        assert isinstance(topology, list)
        assert len(topology) > 0

    def test_layer_topology_structure(self, fitted_model):
        """Test structure of layer topology."""
        topology = fitted_model.get_layer_topology()

        for layer_info in topology:
            assert isinstance(layer_info, dict)
            assert "input" in layer_info
            assert "output" in layer_info


class TestModelSerialization:
    """Test model checkpoint saving/loading."""

    @pytest.fixture
    def fitted_model(self):
        """Create fitted model."""
        model = MNISTLinearSmall()
        data = torch.randn(32, 28, 28)
        model.fit_encoder(data)
        return model

    def test_save_checkpoint(self, fitted_model):
        """Test saving model checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            fitted_model.save_checkpoint(path)

            assert path.exists()

    def test_load_checkpoint(self, fitted_model):
        """Test loading model checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            fitted_model.save_checkpoint(path)

            # Create new model and load
            new_model = MNISTLinearSmall()
            new_model.fit_encoder(torch.randn(32, 28, 28))  # Fit encoder first
            new_model.load_checkpoint(path)

            assert new_model.encoder_fitted == fitted_model.encoder_fitted


class TestRegistryIntegration:
    """Test integration with global REGISTRY."""

    def test_models_can_be_retrieved_from_registry(self):
        """Test that models can be retrieved from REGISTRY (if registered)."""
        # Note: Models in models/__init__.py use get_model() not REGISTRY
        # This test verifies the pattern works with REGISTRY
        models = REGISTRY.list_models()
        assert isinstance(models, list)


class TestModelMetadata:
    """Test model metadata and information."""

    def test_model_name_attribute(self):
        """Test that models have name attribute."""
        model = MNISTLinearSmall()
        assert hasattr(model, "model_name")
        assert model.model_name == "mnist_fc_8k_linear"

    def test_model_repr(self):
        """Test model string representation."""
        model = MNISTLinearSmall()
        data = torch.randn(32, 28, 28)
        model.fit_encoder(data)

        repr_str = repr(model)
        assert "mnist_fc_8k_linear" in repr_str
        assert "input_size" in repr_str
        assert "num_classes" in repr_str


class TestVariantModels:
    """Test variant models (bit flipping, grad norm, etc.)."""

    def test_bitflip_variants_exist(self):
        """Test that all bit flip variants are available."""
        assert "mnist_bitflip_none" in list_models()
        assert "mnist_bitflip_5" in list_models()
        assert "mnist_bitflip_10" in list_models()
        assert "mnist_bitflip_20" in list_models()

    def test_gradnorm_variants_exist(self):
        """Test that all gradient normalization variants are available."""
        assert "mnist_gradnorm_none" in list_models()
        assert "mnist_gradnorm_layerwise" in list_models()
        assert "mnist_gradnorm_batchwise" in list_models()

    def test_bitflip_variant_instantiation(self):
        """Test instantiation of bit flip variants."""
        for name in ["mnist_bitflip_none", "mnist_bitflip_5", "mnist_bitflip_10"]:
            model = get_model(name)
            assert isinstance(model, BaseModel)

    def test_gradnorm_variant_instantiation(self):
        """Test instantiation of gradient norm variants."""
        for name in ["mnist_gradnorm_none", "mnist_gradnorm_layerwise"]:
            model = get_model(name)
            assert isinstance(model, BaseModel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
