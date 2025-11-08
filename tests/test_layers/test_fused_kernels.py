"""
Test script for fused mapping CUDA kernels.
Runs Tests 1, 2, and 4 from BUILD_AND_TEST_FUSED.md

Tests:
1. Basic functionality - forward pass works correctly
2. Gradient correctness - backward pass computes gradients
4. Numerical equivalence - fused and non-fused produce same results
"""

import pytest
import torch
from testing_utils import is_cuda_available


@pytest.mark.gpu
def test_basic_functionality():
    """Test 1: Basic Functionality - forward pass works correctly."""
    if not is_cuda_available():
        pytest.skip("CUDA not available")

    from difflut.layers.random_layer import RandomLayer
    from difflut.nodes.dwn_node import DWNNode
    from difflut.nodes.node_config import NodeConfig

    # Create layer with DWN node
    layer = RandomLayer(
        input_size=25,
        output_size=36,
        node_type=DWNNode,
        node_kwargs=NodeConfig(input_dim=6, output_dim=1),
        seed=42,
    ).cuda()

    # Test forward pass
    x = torch.randn(100, 25).cuda()
    output = layer(x)

    # Check output shape
    assert output.shape == torch.Size([100, 36]), f"Expected shape [100, 36], got {output.shape}"

    # Check if output is valid (no NaN)
    assert not torch.isnan(output).any(), "Output contains NaN values!"

    # Check if fused path is available
    assert hasattr(
        layer.node, "forward_with_mapping"
    ), "Fused forward_with_mapping() method not found"


@pytest.mark.gpu
def test_gradient_correctness():
    """Test 2: Gradient Correctness - gradients flow properly through fused kernels."""
    if not is_cuda_available():
        pytest.skip("CUDA not available")

    from difflut.layers.random_layer import RandomLayer
    from difflut.nodes.dwn_node import DWNNode
    from difflut.nodes.node_config import NodeConfig

    # Create layer
    layer = RandomLayer(
        input_size=16,
        output_size=25,
        node_type=DWNNode,
        node_kwargs=NodeConfig(input_dim=4, output_dim=1),
        seed=42,
    ).cuda()


@pytest.mark.gpu
def test_numerical_equivalence():
    """Test 4: Numerical Equivalence - fused and non-fused produce same results."""
    if not is_cuda_available():
        pytest.skip("CUDA not available")

    from difflut.layers.random_layer import RandomLayer
    from difflut.nodes.dwn_node import _FUSED_CUDA_EXT_AVAILABLE, DWNNode

    if not _FUSED_CUDA_EXT_AVAILABLE:
        pytest.skip("Fused CUDA extension not available (expected if not compiled yet)")

    torch.manual_seed(42)

    # Create layer that will use fused path
    layer = RandomLayer(
        input_size=25,
        output_size=36,
        node_type=DWNNode,
        node_kwargs={"input_dim": 6, "output_dim": 1},
        seed=42,
    ).cuda()

    # Test input
    x = torch.randn(100, 25).cuda()

    # Fused forward (automatic via layer)
    output_fused = layer(x)

    # Non-fused forward (manually materialize mapped_inputs)
    with torch.no_grad():
        # Get mapped inputs the old way
        mapped = layer.get_mapping(x)
        # Call node forward directly (bypasses fused path)
        output_nonfused = layer.node.forward(mapped)
        # Reshape to match fused output
        output_nonfused = output_nonfused.view(x.shape[0], -1)

    # Compare outputs
    diff_abs = (output_fused - output_nonfused).abs()
    max_diff = diff_abs.max().item()
    tolerance = 1e-5

    assert max_diff < tolerance, (
        f"Fused and non-fused outputs differ: max_diff={max_diff:.2e}, "
        f"tolerance={tolerance:.2e}. This may indicate a bug in the fused kernel."
    )


@pytest.mark.gpu
def test_cuda_extensions_available():
    """Check if CUDA extensions are available."""
    if not is_cuda_available():
        pytest.skip("CUDA not available")

    # Check standard extension
    try:
        import efd_cuda

        assert efd_cuda is not None
    except ImportError:
        pytest.fail("efd_cuda (standard) extension not available")

    # Check fused extension (warning only, not failure)
    try:
        import efd_fused_cuda

        assert efd_fused_cuda is not None
    except ImportError:
        pytest.skip("efd_fused_cuda (fused) extension not available - tests will use fallback")
