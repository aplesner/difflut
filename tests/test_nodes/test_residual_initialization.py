"""
Comprehensive tests for residual initialization across all node types.

Tests verify:
1. Perfect residual: With noise_factor=0, nodes pass through first input perfectly
2. Training behavior: During forward_train, output depends only on first input
3. Noisy residual: With noise_factor>0, other weights have noise
"""

import itertools

import pytest
import torch

from difflut.nodes import (
    DWNNode,
    DWNStableNode,
    HybridNode,
    LinearLUTNode,
    PolyLUTNode,
    ProbabilisticNode,
)
from difflut.nodes.utils.initializers import residual_init


def _get_node_type_for_init(node_class):
    """Convert node class name to the node_type string for residual_init."""
    name = node_class.__name__
    if name == "DWNStableNode":
        return "dwn_stable"
    elif name == "LinearLUTNode":
        return "linear_lut"
    elif name == "PolyLUTNode":
        return "polylut"
    elif name == "NeuralLUTNode":
        return "neurallut"
    else:
        # Default: lowercase and remove "Node" suffix
        return name.lower().replace("node", "")


class TestResidualInitPerfectPass:
    """Test perfect residual initialization (noise_factor=0)."""

    @pytest.mark.parametrize(
        "node_class,node_kwargs",
        [
            (DWNNode, {"input_dim": 6, "output_dim": 1}),
            (DWNStableNode, {"input_dim": 6, "output_dim": 1}),
            (ProbabilisticNode, {"input_dim": 6, "output_dim": 1}),
            (HybridNode, {"input_dim": 6, "output_dim": 1}),
            (LinearLUTNode, {"input_dim": 6, "output_dim": 1}),
            (PolyLUTNode, {"input_dim": 4, "output_dim": 1, "degree": 2}),
        ],
    )
    def test_perfect_residual_eval_mode(self, node_class, node_kwargs, device):
        """
        Test that forward_eval with noise_factor=0 gives perfect pass-through.

        In eval mode, output should exactly equal the first input for all test cases.
        """
        input_dim = node_kwargs["input_dim"]

        # Build init_kwargs with node_type and input_dim
        node_type = _get_node_type_for_init(node_class)
        init_kwargs = {
            "node_type": node_type,
            "noise_factor": 0.0,
            "logit_clarity": 5.0,
            "input_dim": input_dim,  # Required for LUT-based nodes
        }

        # Add node-specific parameters
        if node_type == "polylut":
            # Will be set by PolyLUT
            init_kwargs["monomial_combinations"] = None

        node = node_class(
            init_fn=residual_init,
            init_kwargs=init_kwargs,
            **node_kwargs,
        ).to(device)
        node.eval()

        # Generate all binary combinations for comprehensive testing
        # For PolyLUT with input_dim=4, generate all 2^4=16 combinations
        # For other nodes with input_dim=6, generate a subset (all 2^6=64 combinations)
        all_inputs = []
        max_bits = min(input_dim, 6)
        for bits in itertools.product([0, 1], repeat=max_bits):
            all_inputs.append(bits + (0,) * (input_dim - max_bits))

        all_inputs = torch.tensor(all_inputs, dtype=torch.float32, device=device)

        # Extract first input (bit 0, which is idx & 1 in LUT indexing)
        first_input = all_inputs[:, 0]

        with torch.no_grad():
            output = node.forward_eval(all_inputs).squeeze()

        # Check perfect pass-through in eval mode
        accuracy = (output == first_input).float().mean().item()
        assert accuracy == 1.0, (
            f"{node_class.__name__} eval mode accuracy: {accuracy:.4f} "
            f"(expected 1.0 for perfect residual)"
        )

    @pytest.mark.parametrize(
        "node_class,node_kwargs",
        [
            (DWNNode, {"input_dim": 6, "output_dim": 1}),
            (DWNStableNode, {"input_dim": 6, "output_dim": 1}),
            (ProbabilisticNode, {"input_dim": 6, "output_dim": 1}),
            (HybridNode, {"input_dim": 6, "output_dim": 1}),
            (LinearLUTNode, {"input_dim": 6, "output_dim": 1}),
            (PolyLUTNode, {"input_dim": 4, "output_dim": 1, "degree": 2}),
        ],
    )
    def test_perfect_residual_train_mode_first_input_dependent(
        self, node_class, node_kwargs, device
    ):
        """
        Test that forward_train with noise_factor=0 depends only on first input.

        In training mode with sigmoid, the output should strongly correlate with
        the first input. We verify this by checking that:
        - When first_input=1: output ≈ sigmoid(5) ≈ 0.993
        - When first_input=0: output ≈ sigmoid(-5) ≈ 0.007
        """
        input_dim = node_kwargs["input_dim"]

        # Build init_kwargs with node_type and input_dim
        node_type = _get_node_type_for_init(node_class)
        init_kwargs = {
            "node_type": node_type,
            "noise_factor": 0.0,
            "logit_clarity": 5.0,
            "input_dim": input_dim,  # Required for LUT-based nodes
        }

        # Add node-specific parameters
        if node_type == "polylut":
            # Will be set by PolyLUT
            init_kwargs["monomial_combinations"] = None

        node = node_class(
            init_fn=residual_init,
            init_kwargs=init_kwargs,
            **node_kwargs,
        ).to(device)
        node.train()

        # Generate test inputs where only first bit varies
        first_bit_0 = torch.zeros((50, input_dim), device=device)
        first_bit_1 = torch.ones((50, input_dim), device=device)

        with torch.no_grad():
            output_first_0 = node.forward_train(first_bit_0).squeeze()
            output_first_1 = node.forward_train(first_bit_1).squeeze()

        # Expected sigmoid outputs
        sigmoid_5 = torch.sigmoid(torch.tensor(5.0)).item()  # ≈ 0.993
        sigmoid_minus5 = torch.sigmoid(torch.tensor(-5.0)).item()  # ≈ 0.007

        # Check that outputs are close to expected sigmoid values
        # Allow some tolerance due to node-specific implementations
        mean_output_0 = output_first_0.mean().item()
        mean_output_1 = output_first_1.mean().item()

        # For nodes with sigmoid (not DWN which uses raw values)
        if node_class.__name__ != "DWNNode":
            assert (
                abs(mean_output_0 - sigmoid_minus5) < 0.1
            ), f"{node_class.__name__}: output for first_input=0 is {mean_output_0:.4f}, expected ≈{sigmoid_minus5:.4f}"
            assert (
                abs(mean_output_1 - sigmoid_5) < 0.1
            ), f"{node_class.__name__}: output for first_input=1 is {mean_output_1:.4f}, expected ≈{sigmoid_5:.4f}"

    @pytest.mark.parametrize(
        "node_class,node_kwargs",
        [
            (DWNStableNode, {"input_dim": 6, "output_dim": 1}),
            (ProbabilisticNode, {"input_dim": 6, "output_dim": 1}),
            (HybridNode, {"input_dim": 6, "output_dim": 1}),
            (LinearLUTNode, {"input_dim": 6, "output_dim": 1}),
            (PolyLUTNode, {"input_dim": 4, "output_dim": 1, "degree": 2}),
        ],
    )
    def test_perfect_residual_first_input_only_dependency(self, node_class, node_kwargs, device):
        """
        Test that with noise_factor=0, output depends ONLY on first input.

        Create pairs of inputs that differ only in bits 1..n (not bit 0).
        If output is truly dependent only on first input, both should give
        identical outputs.
        """
        input_dim = node_kwargs["input_dim"]

        # Build init_kwargs with node_type and input_dim
        node_type = _get_node_type_for_init(node_class)
        init_kwargs = {
            "node_type": node_type,
            "noise_factor": 0.0,
            "logit_clarity": 5.0,
            "input_dim": input_dim,  # Required for LUT-based nodes
        }

        # Add node-specific parameters
        if node_type == "polylut":
            # Will be set by PolyLUT
            init_kwargs["monomial_combinations"] = None

        node = node_class(
            init_fn=residual_init,
            init_kwargs=init_kwargs,
            **node_kwargs,
        ).to(device)
        node.train()

        # Create two inputs that only differ in non-first bits
        # Input A: first_bit=0, others=0
        input_a = torch.zeros((1, input_dim), device=device)

        # Input B: first_bit=0, others=1 (different in all non-first bits)
        input_b = torch.zeros((1, input_dim), device=device)
        input_b[0, 1:] = 1.0

        with torch.no_grad():
            output_a = node.forward_train(input_a).squeeze()
            output_b = node.forward_train(input_b).squeeze()

        # Outputs should be identical (differ only in other bits, but output
        # depends only on first bit with noise_factor=0)
        diff = abs(output_a.item() - output_b.item())
        assert (
            diff < 1e-5
        ), f"{node_class.__name__}: outputs differ by {diff:.6f} for identical first bits with noise_factor=0"


class TestResidualInitNoisyResidual:
    """Test noisy residual initialization (noise_factor > 0)."""

    @pytest.mark.parametrize(
        "node_class,node_kwargs",
        [
            (DWNNode, {"input_dim": 6, "output_dim": 1}),
            (DWNStableNode, {"input_dim": 6, "output_dim": 1}),
            (ProbabilisticNode, {"input_dim": 6, "output_dim": 1}),
            (HybridNode, {"input_dim": 6, "output_dim": 1}),
            (LinearLUTNode, {"input_dim": 6, "output_dim": 1}),
            (PolyLUTNode, {"input_dim": 4, "output_dim": 1, "degree": 2}),
        ],
    )
    def test_noisy_residual_has_noise_on_other_weights(self, node_class, node_kwargs, device):
        """
        Test that with noise_factor > 0, other weights are noisy.

        Create multiple nodes with the same noise_factor and verify they
        have different parameter values (indicating noise was applied).
        """
        noise_factor = 0.1
        input_dim = node_kwargs["input_dim"]

        # Build init_kwargs with node_type and input_dim
        node_type = _get_node_type_for_init(node_class)
        init_kwargs = {
            "node_type": node_type,
            "noise_factor": noise_factor,
            "logit_clarity": 5.0,
            "input_dim": input_dim,  # Required for LUT-based nodes
        }

        # Add node-specific parameters
        if node_type == "polylut":
            # Will be set by PolyLUT
            init_kwargs["monomial_combinations"] = None

        # Create multiple nodes with same noise_factor
        nodes = []
        for _ in range(3):
            node = node_class(
                init_fn=residual_init,
                init_kwargs=init_kwargs.copy(),
                **node_kwargs,
            ).to(device)
            nodes.append(node)

        # Get parameters from each node
        params_list = []
        for node in nodes:
            for param in node.parameters():
                params_list.append(param.data.clone())

        # Check that parameters are different (noisy, not deterministic)
        # Compare first and second node
        diff = (params_list[0] - params_list[1]).abs().max().item()
        assert (
            diff > 1e-6
        ), f"{node_class.__name__}: parameters identical with noise_factor={noise_factor}, suggesting no noise applied"

    @pytest.mark.parametrize(
        "node_class,node_kwargs",
        [
            (DWNStableNode, {"input_dim": 6, "output_dim": 1}),
            (ProbabilisticNode, {"input_dim": 6, "output_dim": 1}),
            (HybridNode, {"input_dim": 6, "output_dim": 1}),
            (LinearLUTNode, {"input_dim": 6, "output_dim": 1}),
            (PolyLUTNode, {"input_dim": 4, "output_dim": 1, "degree": 2}),
        ],
    )
    def test_noisy_residual_maintains_correlation(self, node_class, node_kwargs, device):
        """
        Test that with noise_factor > 0, first input still has high correlation with output.

        Even with noise on other weights, the initialization should ensure
        the first input is still the dominant factor.
        """
        noise_factor = 0.1
        input_dim = node_kwargs["input_dim"]

        # Build init_kwargs with node_type and input_dim
        node_type = _get_node_type_for_init(node_class)
        init_kwargs = {
            "node_type": node_type,
            "noise_factor": noise_factor,
            "logit_clarity": 5.0,
            "input_dim": input_dim,  # Required for LUT-based nodes
        }

        # Add node-specific parameters
        if node_type == "polylut":
            # Will be set by PolyLUT
            init_kwargs["monomial_combinations"] = None

        node = node_class(
            init_fn=residual_init,
            init_kwargs=init_kwargs,
            **node_kwargs,
        ).to(device)
        node.eval()

        batch_size = 100

        # Generate test inputs with varying first bit
        test_inputs = []
        for first_bit in [0, 1]:
            for _ in range(batch_size // 2):
                inp = torch.randint(0, 2, (1, input_dim), device=device).float()
                inp[0, 0] = first_bit
                test_inputs.append(inp)

        test_inputs = torch.cat(test_inputs, dim=0)
        first_input = test_inputs[:, 0]

        with torch.no_grad():
            output = node.forward_eval(test_inputs).squeeze()

        # Calculate accuracy (should still be high even with noise)
        accuracy = (output == first_input).float().mean().item()
        assert (
            accuracy >= 0.5
        ), f"{node_class.__name__}: accuracy {accuracy:.4f} too low with noise_factor={noise_factor}"


class TestResidualInitParameterPropagation:
    """Test that residual initialization parameters propagate correctly."""

    def test_noise_factor_parameter_propagation(self, device):
        """Test that noise_factor parameter affects initialization."""
        node_kwargs = {"input_dim": 6, "output_dim": 1}

        # Node with no noise
        node_clean = DWNStableNode(
            init_fn=residual_init,
            init_kwargs={
                "node_type": "dwn_stable",
                "noise_factor": 0.0,
                "logit_clarity": 5.0,
                "input_dim": 6,  # Required for DWNStable
            },
            **node_kwargs,
        ).to(device)

        # Node with noise
        node_noisy = DWNStableNode(
            init_fn=residual_init,
            init_kwargs={
                "node_type": "dwn_stable",
                "noise_factor": 0.2,
                "logit_clarity": 5.0,
                "input_dim": 6,  # Required for DWNStable
            },
            **node_kwargs,
        ).to(device)

        # Get parameter values
        params_clean = [p.data.clone() for p in node_clean.parameters()]
        params_noisy = [p.data.clone() for p in node_noisy.parameters()]

        # With noise_factor=0.0, params should be more uniform
        # With noise_factor>0.0, params should have higher variance
        # Suppress variance warning for single-element tensors
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*var.*degrees of freedom.*")
            var_clean = sum(p.var().item() for p in params_clean) / len(params_clean)
            var_noisy = sum(p.var().item() for p in params_noisy) / len(params_noisy)

        # Noisy should have similar or higher variance (noise adds variance)
        # Note: This depends on the parameter range, so we just check they're different
        assert (
            var_clean != var_noisy
        ), "Clean and noisy nodes should have different parameter variance"

    def test_logit_clarity_parameter_propagation(self, device):
        """Test that logit_clarity parameter affects initialization."""
        node_kwargs = {"input_dim": 6, "output_dim": 1}

        # Node with low clarity
        node_low = LinearLUTNode(
            init_fn=residual_init,
            init_kwargs={
                "node_type": "linear_lut",
                "noise_factor": 0.0,
                "logit_clarity": 1.0,
            },
            **node_kwargs,
        ).to(device)

        # Node with high clarity
        node_high = LinearLUTNode(
            init_fn=residual_init,
            init_kwargs={
                "node_type": "linear_lut",
                "noise_factor": 0.0,
                "logit_clarity": 10.0,
            },
            **node_kwargs,
        ).to(device)

        # Get first weight (should be set to 2*logit_clarity)
        weight_low = node_low.weights[0, 0].item()
        weight_high = node_high.weights[0, 0].item()

        # Weight should be proportional to 2*logit_clarity
        assert abs(weight_low - 2.0) < 0.1, f"Expected weight ≈ 2.0, got {weight_low:.4f}"
        assert abs(weight_high - 20.0) < 0.1, f"Expected weight ≈ 20.0, got {weight_high:.4f}"

        # Check bias is set to -logit_clarity
        bias_low = node_low.bias[0].item()
        bias_high = node_high.bias[0].item()

        assert abs(bias_low - (-1.0)) < 0.1, f"Expected bias ≈ -1.0, got {bias_low:.4f}"
        assert abs(bias_high - (-10.0)) < 0.1, f"Expected bias ≈ -10.0, got {bias_high:.4f}"


class TestResidualInitPolynomial:
    """Specific tests for polynomial (PolyLUT) residual initialization."""

    def test_polynomial_first_input_linear_term(self, device):
        """Test that PolyLUT initializes first input linear term correctly."""
        node = PolyLUTNode(
            input_dim=4,
            output_dim=1,
            degree=2,
            init_fn=residual_init,
            init_kwargs={
                "node_type": "polylut",
                "noise_factor": 0.0,
                "logit_clarity": 5.0,
            },
        ).to(device)

        # Find the (1,0,0,0) monomial (first input linear term)
        first_input_linear_idx = None
        for idx, exponents in enumerate(node.monomial_combinations):
            if exponents[0] == 1 and all(e == 0 for e in exponents[1:]):
                first_input_linear_idx = idx
                break

        assert first_input_linear_idx is not None, "Could not find (1,0,0,0) monomial"

        # Check that this coefficient is set to 2*logit_clarity (for offset sigmoid)
        coeff = node.weights[first_input_linear_idx, 0].item()
        assert abs(coeff - 10.0) < 0.1, f"Expected coefficient ≈ 10.0, got {coeff:.4f}"

    def test_polynomial_other_terms_zero_noise(self, device):
        """Test that PolyLUT initializes other terms appropriately (with noise_factor=0)."""
        node = PolyLUTNode(
            input_dim=4,
            output_dim=1,
            degree=2,
            init_fn=residual_init,
            init_kwargs={
                "node_type": "polylut",
                "noise_factor": 0.0,
                "logit_clarity": 5.0,
            },
        ).to(device)

        # Find the (1,0,0,0) monomial index and (0,0,0,0) constant index
        first_input_linear_idx = None
        constant_idx = None
        for idx, exponents in enumerate(node.monomial_combinations):
            if exponents[0] == 1 and all(e == 0 for e in exponents[1:]):
                first_input_linear_idx = idx
            elif all(e == 0 for e in exponents):
                constant_idx = idx

        # Check constant term is -logit_clarity
        if constant_idx is not None:
            coeff = node.weights[constant_idx, 0].item()
            assert (
                abs(coeff - (-5.0)) < 1e-5
            ), f"Expected constant coefficient ≈ -5.0, got {coeff:.4f}"

        # Check that all other coefficients are zero (noise_factor=0)
        for idx in range(len(node.monomial_combinations)):
            if idx != first_input_linear_idx and idx != constant_idx:
                coeff = node.weights[idx, 0].item()
                assert (
                    abs(coeff) < 1e-5
                ), f"Expected coefficient ≈ 0, got {coeff:.4f} at index {idx}"
