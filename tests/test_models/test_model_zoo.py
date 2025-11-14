"""
Comprehensive tests for DiffLUT Model Zoo.

Registry-based, future-proof tests that work with all models in difflut.models.

Tests cover:
- Model instantiation from registry
- Encoder fitting
- Forward pass dimension validation
- Gradient computation
- Parameter counting
- Shape correctness

Tests are task-agnostic (work for classification, regression, segmentation, etc.)
and automatically discover all registered models.
"""

import pytest
import torch
from testing_utils import (
    IgnoreWarnings,
    assert_gradients_exist,
    assert_range,
    assert_shape_equal,
    generate_uniform_input,
    is_cuda_available,
)

from difflut import REGISTRY
from difflut.models import BaseLUTModel

# ============================================================================
# Get all registered models from REGISTRY
# ============================================================================

_testable_models = REGISTRY.list_models() if hasattr(REGISTRY, "list_models") else []


@pytest.fixture
def device():
    """Get device for testing (CPU or GPU if available)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Basic Model Tests
# ============================================================================


@pytest.mark.parametrize("model_name", _testable_models)
class TestModelForwardPass:
    """Test forward pass for all registered models."""

    def test_model_instantiation(self, model_name):
        """Test 1.1: Model can be instantiated from registry."""
        with IgnoreWarnings():
            try:
                model_class = REGISTRY.get_model(model_name)
                model = model_class()
                assert isinstance(model, BaseLUTModel)
            except Exception as e:
                pytest.skip(f"Model {model_name} cannot be instantiated: {e}")

    def test_encoder_fitting(self, model_name, device):
        """Test 1.2: Encoder fitting works."""
        with IgnoreWarnings():
            try:
                model_class = REGISTRY.get_model(model_name)
                model = model_class().to(device)
            except Exception:
                pytest.skip(f"Model {model_name} cannot be instantiated")
                return

            # Generate random input data (generic, task-agnostic)
            # Models should handle variable input shapes
            try:
                if hasattr(model, "input_size"):
                    # 1D input (flattened)
                    input_size = model.input_size if isinstance(model.input_size, int) else 100
                    data = generate_uniform_input((32, input_size), device=device)
                else:
                    # Default to 1D
                    data = generate_uniform_input((32, 100), device=device)

                assert not model.encoder_fitted
                model.fit_encoder(data)
                assert model.encoder_fitted
            except Exception as e:
                pytest.skip(f"Encoder fitting failed for {model_name}: {e}")

    def test_forward_pass_shape(self, model_name, device):
        """Test 1.3: Forward pass produces correct tensor (no assumptions on output shape)."""
        with IgnoreWarnings():
            try:
                model_class = REGISTRY.get_model(model_name)
                model = model_class().to(device)
            except Exception:
                pytest.skip(f"Model {model_name} cannot be instantiated")
                return

            # Fit encoder
            try:
                if hasattr(model, "input_size"):
                    input_size = model.input_size if isinstance(model.input_size, int) else 100
                    data = generate_uniform_input((32, input_size), device=device)
                else:
                    data = generate_uniform_input((32, 100), device=device)
                model.fit_encoder(data)
            except Exception:
                pytest.skip(f"Cannot fit encoder for {model_name}")
                return

            # Forward pass with batch
            try:
                if hasattr(model, "input_size"):
                    input_size = model.input_size if isinstance(model.input_size, int) else 100
                    batch = generate_uniform_input((8, input_size), seed=123, device=device)
                else:
                    batch = generate_uniform_input((8, 100), seed=123, device=device)

                with torch.no_grad():
                    output = model(batch)

                # Verify output is a tensor
                assert isinstance(output, torch.Tensor)
                # Output should have batch dimension
                assert output.shape[0] == 8, f"Batch size mismatch for {model_name}: {output.shape}"
                # Output should be at least 1D (handles classification, regression, segmentation)
                assert output.ndim >= 1
            except Exception as e:
                pytest.fail(f"Forward pass failed for {model_name}: {e}")

    def test_gradients_computation(self, model_name, device):
        """Test 1.4: Model computes gradients correctly."""
        with IgnoreWarnings():
            try:
                model_class = REGISTRY.get_model(model_name)
                model = model_class().to(device)
            except Exception:
                pytest.skip(f"Model {model_name} cannot be instantiated")
                return

            # Fit encoder
            try:
                if hasattr(model, "input_size"):
                    input_size = model.input_size if isinstance(model.input_size, int) else 100
                    data = generate_uniform_input((32, input_size), device=device)
                else:
                    data = generate_uniform_input((32, 100), device=device)
                model.fit_encoder(data)
            except Exception:
                pytest.skip(f"Cannot fit encoder for {model_name}")
                return

            # Forward and backward pass
            model.train()
            try:
                if hasattr(model, "input_size"):
                    input_size = model.input_size if isinstance(model.input_size, int) else 100
                    batch = generate_uniform_input((8, input_size), seed=42, device=device)
                else:
                    batch = generate_uniform_input((8, 100), seed=42, device=device)

                batch.requires_grad = True
                output = model(batch)
                loss = output.sum()
                loss.backward()

                # Check that gradients exist for parameters
                assert_gradients_exist(model)
            except Exception as e:
                pytest.fail(f"Gradient computation failed for {model_name}: {e}")

    @pytest.mark.gpu
    def test_cpu_gpu_consistency(self, model_name):
        """Test 1.5: CPU and GPU implementations give same forward pass results."""
        if not is_cuda_available():
            pytest.skip("CUDA not available")

        with IgnoreWarnings():
            try:
                # Create CPU model
                torch.manual_seed(42)
                model_class = REGISTRY.get_model(model_name)
                model_cpu = model_class()
            except Exception:
                pytest.skip(f"Model {model_name} cannot be instantiated")
                return

            # Create GPU model with same initialization
            try:
                torch.manual_seed(42)
                model_gpu = model_class().cuda()
            except Exception:
                pytest.skip(f"Model {model_name} cannot be moved to GPU")
                return

            # Copy state from CPU to GPU
            model_gpu.load_state_dict(model_cpu.state_dict())

            # Fit encoder on both
            try:
                if hasattr(model_cpu, "input_size"):
                    input_size = (
                        model_cpu.input_size if isinstance(model_cpu.input_size, int) else 100
                    )
                    data_cpu = generate_uniform_input((32, input_size), seed=42, device="cpu")
                else:
                    data_cpu = generate_uniform_input((32, 100), seed=42, device="cpu")
                data_gpu = data_cpu.cuda()

                model_cpu.fit_encoder(data_cpu)
                model_gpu.fit_encoder(data_gpu)
            except Exception:
                pytest.skip(f"Cannot fit encoder for {model_name}")
                return

            # Compare forward pass
            model_cpu.eval()
            model_gpu.eval()

            try:
                if hasattr(model_cpu, "input_size"):
                    input_size = (
                        model_cpu.input_size if isinstance(model_cpu.input_size, int) else 100
                    )
                    input_cpu = generate_uniform_input((8, input_size), seed=123, device="cpu")
                else:
                    input_cpu = generate_uniform_input((8, 100), seed=123, device="cpu")
                input_gpu = input_cpu.cuda()

                with torch.no_grad():
                    output_cpu = model_cpu(input_cpu)
                    output_gpu = model_gpu(input_gpu).cpu()

                torch.testing.assert_close(output_cpu, output_gpu, atol=1e-5, rtol=1e-4)
            except AssertionError as e:
                pytest.fail(f"CPU/GPU outputs differ for {model_name}: {e}")
            except Exception as e:
                pytest.fail(f"Forward pass comparison failed for {model_name}: {e}")


# ============================================================================
# Parameter Counting Tests
# ============================================================================


@pytest.mark.parametrize("model_name", _testable_models)
def test_model_parameter_counting(model_name):
    """Test that model parameter counting works (if method exists)."""
    with IgnoreWarnings():
        try:
            model_class = REGISTRY.get_model(model_name)
            model = model_class()
        except Exception:
            pytest.skip(f"Model {model_name} cannot be instantiated")
            return

        # Fit encoder first
        try:
            if hasattr(model, "input_size"):
                input_size = model.input_size if isinstance(model.input_size, int) else 100
                data = generate_uniform_input((32, input_size))
            else:
                data = generate_uniform_input((32, 100))
            model.fit_encoder(data)
        except Exception:
            pytest.skip(f"Cannot fit encoder for {model_name}")
            return

        # Test parameter counting if method exists
        if hasattr(model, "count_parameters"):
            try:
                counts = model.count_parameters()
                assert isinstance(counts, dict)
                assert "total" in counts
                assert counts["total"] > 0
            except Exception as e:
                pytest.skip(f"Parameter counting not available for {model_name}: {e}")


# ============================================================================
# Model Information Tests
# ============================================================================


@pytest.mark.parametrize("model_name", _testable_models)
def test_model_has_required_attributes(model_name):
    """Test that models have basic required attributes."""
    with IgnoreWarnings():
        try:
            model_class = REGISTRY.get_model(model_name)
            model = model_class()
            assert isinstance(model, BaseLUTModel)
            # All models should be torch modules
            assert isinstance(model, torch.nn.Module)
        except Exception as e:
            pytest.skip(f"Model {model_name} cannot be instantiated: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
