"""Pytest configuration and fixtures for all tests."""

import sys
from pathlib import Path

import pytest
from testing_utils import get_device, is_cuda_available

# Add tests directory to Python path so testing_utils can be imported
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))


@pytest.fixture
def device(request):
    """
    Fixture that provides the appropriate device for testing.

    Behavior:
    - For tests marked with @pytest.mark.gpu:
      * Requires CUDA to be available
      * Skips test if CUDA not available
      * Returns 'cuda'

    - For all other tests:
      * Returns 'cuda' if available, else 'cpu'
      * Tests automatically use GPU when available
      * Falls back to CPU when GPU not available

    Usage:
        def test_something(device):
            model = MyModel().to(device)
            x = torch.randn(10, 5).to(device)
    """
    # Check if test is marked with @pytest.mark.gpu
    gpu_marker = request.node.get_closest_marker("gpu")

    if gpu_marker:
        # Test requires GPU - skip if not available
        if not is_cuda_available():
            pytest.skip("CUDA not available")
        return "cuda"
    else:
        # Test is device-agnostic - use best available device
        return get_device()
