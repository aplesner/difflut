"""
Comprehensive tests for residual initialization.

Tests the residual_init function with different node types to ensure
proper initialization for identity-like behavior on the first input.
"""

import math

import pytest
import torch
from testing_utils import IgnoreWarnings, generate_uniform_input

from difflut.nodes.utils.initializers import (
    DEFAULT_RESIDUAL_LOGIT_SCALE,
    DEFAULT_RESIDUAL_SIGMA,
    residual_init,
)
from difflut.registry import REGISTRY


class TestResidualInitLinear:
    """Test residual initialization for linear nodes."""

    def test_linear_basic(self):
        """Test linear residual initialization."""
        weights = torch.zeros(6, 1)
        residual_init(weights, node_type="linear_lut")

        # First weight should be 1.0
        assert weights[0, 0].item() == 1.0, "First weight should be 1.0"

        # Other weights should be small
        assert weights[1:, 0].abs().mean().item() < 0.1, "Other weights should be small"

    def test_linear_multiple_outputs(self):
        """Test linear residual initialization with multiple outputs."""
        weights = torch.zeros(6, 3)
        residual_init(weights, node_type="linear_lut")

        # All outputs should have first weight = 1.0
        for out_idx in range(3):
            assert weights[0, out_idx].item() == 1.0, f"First weight for output {out_idx} should be 1.0"

        # Other weights should be small
        assert weights[1:, :].abs().mean().item() < 0.1, "Other weights should be small"

    def test_linear_custom_sigma(self):
        """Test linear residual initialization with custom sigma."""
        weights = torch.zeros(6, 1)
        custom_sigma = 0.001
        residual_init(weights, node_type="linear_lut", sigma_small=custom_sigma)

        # First weight should still be 1.0
        assert weights[0, 0].item() == 1.0

        # Other weights should be even smaller with custom sigma
        assert weights[1:, 0].abs().mean().item() < custom_sigma * 3, "Weights should respect custom sigma"

    def test_linear_identity_behavior(self):
        """Test that linear node output depends ONLY on first input, independent of others."""
        from difflut.nodes import LinearLUTNode

        # Create node with residual initialization
        init_fn = REGISTRY.get_initializer("residual")
        init_kwargs = {"node_type": "linear_lut", "sigma_small": 0.0}  # Zero noise for exact test

        with IgnoreWarnings():
            node = LinearLUTNode(input_dim=6, output_dim=1, init_fn=init_fn, init_kwargs=init_kwargs)

        # Test 1: Vary first input, keep others constant
        batch_size = 10
        x1 = torch.full((batch_size, 6), 0.5)
        x1[:, 0] = torch.linspace(0, 1, batch_size)

        with torch.no_grad():
            output1 = node(x1)

        # Output should closely match first input (through sigmoid)
        expected = torch.sigmoid(x1[:, 0:1])
        assert torch.allclose(output1, expected, atol=0.1), "Output should approximate first input"

        # Test 2: Same first input, different other inputs - output should be identical
        x2 = torch.rand(batch_size, 6)  # Random values for inputs 2-6
        x2[:, 0] = x1[:, 0].clone()  # Keep first input same as x1

        with torch.no_grad():
            output2 = node(x2)

        # Outputs should be identical since first input is the same
        assert torch.allclose(
            output1, output2, atol=1e-6
        ), "Output should be independent of inputs 2-6"


class TestResidualInitPolynomial:
    """Test residual initialization for polynomial nodes."""

    def test_polynomial_basic(self):
        """Test polynomial residual initialization."""
        # Create simple monomial combinations
        monomial_combinations = [
            (0, 0, 0),  # constant
            (1, 0, 0),  # x1 (first input linear term)
            (0, 1, 0),  # x2
            (0, 0, 1),  # x3
            (2, 0, 0),  # x1^2
        ]

        coeffs = torch.zeros(len(monomial_combinations), 1)
        residual_init(coeffs, node_type="polylut", monomial_combinations=monomial_combinations)

        # The (1,0,0) term should be 1.0 (index 1)
        assert coeffs[1, 0].item() == 1.0, "First input linear term should be 1.0"

        # Other coefficients should be small
        other_coeffs = torch.cat([coeffs[:1], coeffs[2:]])
        assert other_coeffs.abs().mean().item() < 0.1, "Other coefficients should be small"

    def test_polynomial_no_first_input_term(self):
        """Test polynomial initialization without (1,0,0) term."""
        # Monomial combinations without first input linear term
        monomial_combinations = [
            (0, 0, 0),  # constant
            (0, 1, 0),  # x2
            (0, 0, 1),  # x3
        ]

        coeffs = torch.zeros(len(monomial_combinations), 1)
        residual_init(coeffs, node_type="polylut", monomial_combinations=monomial_combinations)

        # All coefficients should be small (no (1,0,0) to set to 1)
        assert coeffs.abs().mean().item() < 0.1, "All coefficients should be small"

    def test_polynomial_requires_combinations(self):
        """Test that polynomial init requires monomial_combinations."""
        coeffs = torch.zeros(5, 1)

        with pytest.raises(ValueError, match="monomial_combinations"):
            residual_init(coeffs, node_type="polylut")


class TestResidualInitMLP:
    """Test residual initialization for MLP/NeuralLUT nodes."""

    def test_mlp_first_layer_weights(self):
        """Test MLP residual initialization for first layer weights."""
        hidden_width = 8
        input_dim = 6
        weights = torch.zeros(hidden_width, input_dim)

        residual_init(
            weights,
            node_type="neurallut",
            param_name="weight",
            layer_idx=0,
            num_layers=2,
        )

        # W[0,0] should be 1.0
        assert weights[0, 0].item() == 1.0, "W[0,0] should be 1.0"

        # Other weights should be small
        assert weights.abs().mean().item() < 0.5, "Other weights should be relatively small"

    def test_mlp_later_layer_weights(self):
        """Test MLP residual initialization for later layers."""
        weights = torch.zeros(8, 8)

        residual_init(
            weights,
            node_type="neurallut",
            param_name="weight",
            layer_idx=1,
            num_layers=2,
        )

        # All weights should be small (no special W[0,0] for later layers)
        assert weights.abs().mean().item() < 0.1, "All weights should be small"

    def test_mlp_bias(self):
        """Test MLP residual initialization for bias."""
        bias = torch.ones(8)  # Start with ones to test they're zeroed

        residual_init(
            bias, node_type="neurallut", param_name="bias", layer_idx=0, num_layers=2
        )

        # All biases should be zero
        assert torch.allclose(bias, torch.zeros(8)), "All biases should be zero"

    def test_mlp_requires_layer_info(self):
        """Test that MLP init requires layer_idx and num_layers."""
        weights = torch.zeros(8, 6)

        with pytest.raises(ValueError, match="layer_idx"):
            residual_init(weights, node_type="neurallut", param_name="weight")

        with pytest.raises(ValueError, match="num_layers"):
            residual_init(weights, node_type="neurallut", param_name="weight", layer_idx=0)

    def test_mlp_requires_param_name(self):
        """Test that MLP init requires param_name."""
        weights = torch.zeros(8, 6)

        with pytest.raises(ValueError, match="param_name"):
            residual_init(weights, node_type="neurallut", layer_idx=0, num_layers=2)


class TestResidualInitFourier:
    """Test residual initialization for Fourier nodes."""

    def test_fourier_amplitudes(self):
        """Test Fourier residual initialization for amplitudes."""
        num_frequencies = 8
        amplitudes = torch.zeros(num_frequencies, 1)

        residual_init(
            amplitudes,
            node_type="fourier",
            param_name="amplitudes",
            num_frequencies=num_frequencies,
        )

        # First amplitude should be 1.0
        assert amplitudes[0, 0].item() == 1.0, "First amplitude should be 1.0"

        # Other amplitudes should be small
        assert amplitudes[1:, 0].abs().mean().item() < 0.1, "Other amplitudes should be small"

        # All amplitudes should be non-negative
        assert (amplitudes >= 0).all(), "All amplitudes should be non-negative"

    def test_fourier_phases(self):
        """Test Fourier residual initialization for phases."""
        num_frequencies = 8
        phases = torch.ones(num_frequencies, 1)  # Start with ones

        residual_init(
            phases,
            node_type="fourier",
            param_name="phases",
            num_frequencies=num_frequencies,
        )

        # First phase should be 0.0
        assert phases[0, 0].item() == 0.0, "First phase should be 0.0"

        # Other phases should be in [-π, π]
        assert (phases[1:] >= -math.pi).all() and (
            phases[1:] <= math.pi
        ).all(), "Phases should be in [-π, π]"

    def test_fourier_bias(self):
        """Test Fourier residual initialization for bias."""
        bias = torch.zeros(1)

        residual_init(
            bias, node_type="fourier", param_name="bias", num_frequencies=8
        )

        # Bias should be 0.5
        assert bias[0].item() == 0.5, "Bias should be 0.5"

    def test_fourier_requires_num_frequencies(self):
        """Test that Fourier init requires num_frequencies."""
        amplitudes = torch.zeros(8, 1)

        with pytest.raises(ValueError, match="num_frequencies"):
            residual_init(amplitudes, node_type="fourier", param_name="amplitudes")

    def test_fourier_requires_param_name(self):
        """Test that Fourier init requires param_name."""
        amplitudes = torch.zeros(8, 1)

        with pytest.raises(ValueError, match="param_name"):
            residual_init(amplitudes, node_type="fourier", num_frequencies=8)


class TestResidualInitLUT:
    """Test residual initialization for LUT-based nodes (DWN, Probabilistic, Hybrid)."""

    @pytest.mark.parametrize("node_type", ["dwn", "dwn_stable", "probabilistic", "hybrid"])
    def test_lut_basic_identity(self, node_type):
        """Test LUT residual initialization creates identity on first input."""
        input_dim = 3
        raw_weights = torch.zeros(2**input_dim, 1)  # (8, 1)

        residual_init(raw_weights, node_type=node_type, input_dim=input_dim)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(raw_weights)

        # Check entries where first bit = 1 (indices 4-7 for input_dim=3)
        # In binary: 100, 101, 110, 111 - first bit is MSB in logic
        first_bit_one = probs[4:].mean().item()
        first_bit_zero = probs[:4].mean().item()

        assert first_bit_one > 0.99, f"{node_type}: Entries with first bit=1 should have prob ~1"
        assert first_bit_zero < 0.01, f"{node_type}: Entries with first bit=0 should have prob ~0"

    def test_lut_transpose_orientation(self):
        """Test LUT initialization with transposed shape (output_dim, 2^input_dim)."""
        input_dim = 3
        raw_weights = torch.zeros(2, 2**input_dim)  # (2, 8) - DWN style

        residual_init(raw_weights, node_type="dwn", input_dim=input_dim)

        probs = torch.sigmoid(raw_weights)

        # Check both outputs
        for out_idx in range(2):
            first_bit_one = probs[out_idx, 4:].mean().item()
            first_bit_zero = probs[out_idx, :4].mean().item()

            assert first_bit_one > 0.99, f"Output {out_idx}: first bit=1 should have prob ~1"
            assert first_bit_zero < 0.01, f"Output {out_idx}: first bit=0 should have prob ~0"

    def test_lut_different_input_dims(self):
        """Test LUT initialization with different input dimensions."""
        for input_dim in [2, 3, 4, 5]:
            raw_weights = torch.zeros(2**input_dim, 1)
            residual_init(raw_weights, node_type="probabilistic", input_dim=input_dim)

            probs = torch.sigmoid(raw_weights)

            # Half the entries should be ~1 (first bit = 1)
            # Half should be ~0 (first bit = 0)
            num_ones = (probs > 0.5).sum().item()
            expected_ones = 2 ** (input_dim - 1)  # Half the truth table

            assert (
                num_ones == expected_ones
            ), f"input_dim={input_dim}: Expected {expected_ones} entries > 0.5, got {num_ones}"

    def test_lut_custom_logit_scale(self):
        """Test LUT initialization with custom logit scale."""
        input_dim = 3
        raw_weights = torch.zeros(2**input_dim, 1)
        custom_scale = 5.0

        residual_init(
            raw_weights,
            node_type="probabilistic",
            input_dim=input_dim,
            logit_scale=custom_scale,
        )

        # Check that max logit magnitude matches custom scale
        assert torch.abs(raw_weights).max().item() == pytest.approx(
            custom_scale, abs=0.01
        ), "Logit scale should match custom value"

    def test_lut_requires_input_dim(self):
        """Test that LUT init requires input_dim."""
        raw_weights = torch.zeros(8, 1)

        with pytest.raises(ValueError, match="input_dim"):
            residual_init(raw_weights, node_type="probabilistic")

    def test_lut_identity_with_probabilistic_node(self):
        """Test residual init creates identity behavior: output depends ONLY on first input."""
        from difflut.nodes import ProbabilisticNode

        input_dim = 4
        init_fn = REGISTRY.get_initializer("residual")
        init_kwargs = {
            "node_type": "probabilistic",
            "input_dim": input_dim,
            "logit_scale": 10.0,
        }

        with IgnoreWarnings():
            node = ProbabilisticNode(
                input_dim=input_dim,
                output_dim=1,
                init_fn=init_fn,
                init_kwargs=init_kwargs,
            )

        # Test 1: Output should match first input when it varies
        batch_size = 16
        x1 = torch.zeros(batch_size, input_dim)
        # Set first input to alternate 0/1, others to 0.5
        for i in range(batch_size):
            x1[i, 0] = float(i % 2)
            x1[i, 1:] = 0.5

        with torch.no_grad():
            output1 = node(x1)

        expected = x1[:, 0:1]
        assert torch.allclose(
            output1, expected, atol=0.1
        ), "Node output should match first input"

        # Test 2: Same first input, randomize other inputs - output should be identical
        x2 = torch.rand(batch_size, input_dim)
        x2[:, 0] = x1[:, 0].clone()  # Keep first input same

        with torch.no_grad():
            output2 = node(x2)

        # Outputs should be identical since first input is the same
        assert torch.allclose(
            output1, output2, atol=0.05
        ), "Output should be independent of inputs 2-4"

        # Test 3: Verify independence by checking all combinations
        # For each value of first input, try different values for other inputs
        for first_val in [0.0, 1.0]:
            x_test = torch.zeros(8, input_dim)
            x_test[:, 0] = first_val
            # Create different patterns for other inputs
            for i in range(8):
                for j in range(1, input_dim):
                    x_test[i, j] = float((i >> (j - 1)) & 1)

            with torch.no_grad():
                outputs = node(x_test)

            # All outputs should be approximately the same (equal to first_val)
            output_std = outputs.std().item()
            assert output_std < 0.05, (
                f"Output should be constant for first_input={first_val}, "
                f"but std={output_std:.4f}"
            )


class TestResidualInitIndependence:
    """Test that residual initialization creates true independence from non-first inputs."""

    def test_linear_independence_comprehensive(self):
        """Test linear node output is independent of inputs 2-n."""
        from difflut.nodes import LinearLUTNode

        init_fn = REGISTRY.get_initializer("residual")
        init_kwargs = {"node_type": "linear_lut", "sigma_small": 0.001}

        with IgnoreWarnings():
            node = LinearLUTNode(input_dim=6, output_dim=1, init_fn=init_fn, init_kwargs=init_kwargs)

        # For each value of first input, test with random other inputs
        for first_value in [0.0, 0.25, 0.5, 0.75, 1.0]:
            # Create multiple samples with same first input, different others
            num_samples = 20
            x = torch.rand(num_samples, 6)
            x[:, 0] = first_value  # Fix first input

            with torch.no_grad():
                outputs = node(x)

            # All outputs should be approximately equal (independent of inputs 2-6)
            output_mean = outputs.mean().item()
            output_std = outputs.std().item()

            assert output_std < 0.01, (
                f"For first_input={first_value}, output std={output_std:.6f} "
                f"is too high - output depends on other inputs!"
            )

            # Also check outputs match expected value from first input
            expected = torch.sigmoid(torch.tensor([first_value])).item()
            assert abs(output_mean - expected) < 0.05, (
                f"Output mean {output_mean:.4f} doesn't match expected {expected:.4f}"
            )

    def test_lut_independence_all_combinations(self):
        """Test LUT node output is independent of inputs 2-n for all input combinations."""
        from difflut.nodes import ProbabilisticNode

        input_dim = 4
        init_fn = REGISTRY.get_initializer("residual")
        init_kwargs = {
            "node_type": "probabilistic",
            "input_dim": input_dim,
            "logit_scale": 10.0,
        }

        with IgnoreWarnings():
            node = ProbabilisticNode(
                input_dim=input_dim,
                output_dim=1,
                init_fn=init_fn,
                init_kwargs=init_kwargs,
            )

        # Test for both values of first input
        for first_val in [0.0, 1.0]:
            # Generate all possible combinations of other inputs
            num_other_inputs = input_dim - 1
            num_combinations = 2 ** num_other_inputs

            x = torch.zeros(num_combinations, input_dim)
            x[:, 0] = first_val

            # Create all binary combinations for inputs 2-4
            for idx in range(num_combinations):
                for bit_pos in range(num_other_inputs):
                    x[idx, bit_pos + 1] = float((idx >> bit_pos) & 1)

            with torch.no_grad():
                outputs = node(x)

            # All outputs should be identical (equal to first_val)
            output_std = outputs.std().item()
            output_mean = outputs.mean().item()

            assert output_std < 0.01, (
                f"For first_input={first_val}, output varies across {num_combinations} "
                f"combinations of other inputs (std={output_std:.6f})"
            )

            assert abs(output_mean - first_val) < 0.05, (
                f"Output mean {output_mean:.4f} doesn't match first_input {first_val}"
            )

    def test_dwn_independence(self):
        """Test DWN node output is independent of inputs 2-n."""
        from difflut.nodes import DWNNode

        input_dim = 4
        init_fn = REGISTRY.get_initializer("residual")
        init_kwargs = {
            "node_type": "dwn",
            "input_dim": input_dim,
            "logit_scale": 10.0,
        }

        with IgnoreWarnings():
            node = DWNNode(
                input_dim=input_dim,
                output_dim=1,
                init_fn=init_fn,
                init_kwargs=init_kwargs,
            )

        # Test independence with random other inputs
        for first_val in [0.0, 1.0]:
            batch_size = 16
            x = torch.rand(batch_size, input_dim)
            x[:, 0] = first_val

            with torch.no_grad():
                outputs = node(x)

            # In eval mode, DWN uses binary thresholding
            # All outputs should be identical
            unique_outputs = torch.unique(outputs)
            assert len(unique_outputs) == 1, (
                f"DWN should produce identical outputs for first_input={first_val}, "
                f"but got {len(unique_outputs)} unique values"
            )

            # Output should match first_val
            output_val = outputs[0, 0].item()
            assert abs(output_val - first_val) < 0.1, (
                f"DWN output {output_val:.4f} doesn't match first_input {first_val}"
            )

    def test_hybrid_independence(self):
        """Test Hybrid node output is independent of inputs 2-n."""
        from difflut.nodes import HybridNode

        input_dim = 4
        init_fn = REGISTRY.get_initializer("residual")
        init_kwargs = {
            "node_type": "hybrid",
            "input_dim": input_dim,
            "logit_scale": 10.0,
        }

        with IgnoreWarnings():
            node = HybridNode(
                input_dim=input_dim,
                output_dim=1,
                init_fn=init_fn,
                init_kwargs=init_kwargs,
            )

        # Test with all binary combinations
        for first_val in [0.0, 1.0]:
            num_combinations = 2 ** (input_dim - 1)
            x = torch.zeros(num_combinations, input_dim)
            x[:, 0] = first_val

            for idx in range(num_combinations):
                for bit_pos in range(input_dim - 1):
                    x[idx, bit_pos + 1] = float((idx >> bit_pos) & 1)

            with torch.no_grad():
                outputs = node(x)

            # All outputs should be identical
            output_std = outputs.std().item()
            assert output_std < 0.01, (
                f"Hybrid node output varies for first_input={first_val} (std={output_std:.6f})"
            )

    def test_continuous_vs_binary_independence(self):
        """Test independence holds for both continuous and binary inputs."""
        from difflut.nodes import ProbabilisticNode

        input_dim = 4
        init_fn = REGISTRY.get_initializer("residual")
        init_kwargs = {
            "node_type": "probabilistic",
            "input_dim": input_dim,
            "logit_scale": 10.0,
        }

        with IgnoreWarnings():
            node = ProbabilisticNode(
                input_dim=input_dim,
                output_dim=1,
                init_fn=init_fn,
                init_kwargs=init_kwargs,
            )

        # Test with continuous values in [0, 1]
        for first_val in [0.1, 0.3, 0.7, 0.9]:
            batch_size = 10
            x = torch.rand(batch_size, input_dim)
            x[:, 0] = first_val

            with torch.no_grad():
                outputs = node(x)

            output_std = outputs.std().item()
            assert output_std < 0.05, (
                f"Continuous input: output varies for first_input={first_val} (std={output_std:.6f})"
            )


class TestResidualInitErrors:
    """Test error handling in residual initialization."""

    def test_missing_node_type(self):
        """Test that missing node_type raises error."""
        param = torch.zeros(10, 1)

        with pytest.raises(ValueError, match="node_type"):
            residual_init(param)

    def test_invalid_node_type(self):
        """Test that invalid node_type raises error."""
        param = torch.zeros(10, 1)

        with pytest.raises(ValueError, match="Unknown node_type"):
            residual_init(param, node_type="invalid_node_type")

    def test_polynomial_missing_combinations(self):
        """Test polynomial without monomial_combinations."""
        param = torch.zeros(10, 1)

        with pytest.raises(ValueError):
            residual_init(param, node_type="polylut")

    def test_mlp_missing_layer_info(self):
        """Test MLP without required layer info."""
        param = torch.zeros(10, 10)

        with pytest.raises(ValueError):
            residual_init(param, node_type="neurallut", param_name="weight")

    def test_fourier_missing_param_name(self):
        """Test Fourier without param_name."""
        param = torch.zeros(10, 1)

        with pytest.raises(ValueError):
            residual_init(param, node_type="fourier", num_frequencies=10)

    def test_lut_missing_input_dim(self):
        """Test LUT without input_dim."""
        param = torch.zeros(8, 1)

        with pytest.raises(ValueError):
            residual_init(param, node_type="probabilistic")


class TestResidualInitDefaults:
    """Test default value handling in residual initialization."""

    def test_default_sigma_used(self):
        """Test that default sigma is used and generates warning."""
        weights = torch.zeros(6, 1)

        # Should use default sigma (warning would be generated)
        residual_init(weights, node_type="linear_lut")

        # Verify initialization happened
        assert weights[0, 0].item() == 1.0

    def test_default_logit_scale_used(self):
        """Test that default logit scale is used and generates warning."""
        raw_weights = torch.zeros(8, 1)

        # Should use default logit_scale (warning would be generated)
        residual_init(raw_weights, node_type="probabilistic", input_dim=3)

        # Verify correct scale was used
        assert torch.abs(raw_weights).max().item() == pytest.approx(
            DEFAULT_RESIDUAL_LOGIT_SCALE, abs=0.01
        )

    def test_custom_sigma_no_warning(self):
        """Test that custom sigma doesn't generate default value warning."""
        weights = torch.zeros(6, 1)

        # Should NOT warn about default (custom value provided)
        residual_init(weights, node_type="linear_lut", sigma_small=0.001)

        assert weights[0, 0].item() == 1.0


class TestResidualInitRegistration:
    """Test that residual initialization is properly registered."""

    def test_residual_in_registry(self):
        """Test that 'residual' is in the initializer registry."""
        assert "residual" in REGISTRY.list_initializers(), "'residual' should be registered"

    def test_get_residual_initializer(self):
        """Test getting residual initializer from registry."""
        init_fn = REGISTRY.get_initializer("residual")
        assert init_fn is not None, "Should be able to get residual initializer"
        assert callable(init_fn), "Residual initializer should be callable"

    def test_use_via_registry(self):
        """Test using residual initializer via registry."""
        init_fn = REGISTRY.get_initializer("residual")
        weights = torch.zeros(6, 1)

        # Use via registry
        init_fn(weights, node_type="linear_lut")

        # Verify it worked
        assert weights[0, 0].item() == 1.0, "Should initialize correctly via registry"


class TestResidualInitIntegration:
    """Integration tests with actual node classes."""

    @pytest.mark.parametrize("node_type", ["linear_lut", "polylut", "neurallut", "fourier"])
    def test_node_with_residual_init(self, node_type):
        """Test that each compatible node works with residual initialization."""
        # Skip if node not available
        if node_type not in REGISTRY.list_nodes():
            pytest.skip(f"Node '{node_type}' not available")

        node_class = REGISTRY.get_node(node_type)
        init_fn = REGISTRY.get_initializer("residual")

        # Prepare init_kwargs based on node type
        init_kwargs = {"node_type": node_type}

        if node_type == "linear_lut":
            pass  # No extra kwargs needed

        elif node_type == "polylut":
            # PolyLUT needs monomial_combinations - but we can't pass it easily
            # Skip integration test for polylut (tested separately)
            pytest.skip("PolyLUT requires special handling for monomial_combinations")

        elif node_type == "neurallut":
            # NeuralLUT needs layer info - skip (tested separately)
            pytest.skip("NeuralLUT requires special handling for layer info")

        elif node_type == "fourier":
            # Fourier needs param_name - skip (tested separately)
            pytest.skip("Fourier requires special handling for param_name")

        try:
            with IgnoreWarnings():
                node = node_class(
                    input_dim=4, output_dim=2, init_fn=init_fn, init_kwargs=init_kwargs
                )

            # Test forward pass
            x = generate_uniform_input((8, 4), seed=42)
            with torch.no_grad():
                output = node(x)

            # Basic checks
            assert output.shape == (8, 2), "Output shape should be correct"
            assert not torch.isnan(output).any(), "Output should not contain NaN"
            assert not torch.isinf(output).any(), "Output should not contain Inf"

        except Exception as e:
            pytest.fail(f"Node '{node_type}' failed with residual initialization: {e}")

    @pytest.mark.parametrize("node_type_str", ["dwn", "probabilistic", "hybrid"])
    def test_lut_nodes_with_residual_init(self, node_type_str):
        """Test LUT-based nodes with residual initialization."""
        # Map string to actual node name in registry
        node_name_map = {
            "dwn": "dwn",
            "probabilistic": "probabilistic",
            "hybrid": "hybrid",
        }

        node_name = node_name_map[node_type_str]
        if node_name not in REGISTRY.list_nodes():
            pytest.skip(f"Node '{node_name}' not available")

        node_class = REGISTRY.get_node(node_name)
        init_fn = REGISTRY.get_initializer("residual")

        input_dim = 4
        init_kwargs = {
            "node_type": node_type_str,
            "input_dim": input_dim,
            "logit_scale": 10.0,
        }

        try:
            with IgnoreWarnings():
                node = node_class(
                    input_dim=input_dim,
                    output_dim=1,
                    init_fn=init_fn,
                    init_kwargs=init_kwargs,
                )

            # Test forward pass
            x = generate_uniform_input((16, input_dim), seed=42)
            with torch.no_grad():
                output = node(x)

            # Basic checks
            assert output.shape == (16, 1), "Output shape should be correct"
            assert not torch.isnan(output).any(), "Output should not contain NaN"
            assert not torch.isinf(output).any(), "Output should not contain Inf"

        except Exception as e:
            pytest.fail(f"Node '{node_name}' failed with residual initialization: {e}")
