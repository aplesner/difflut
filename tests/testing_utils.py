"""
Test utilities and fixtures for DiffLUT test suite.
Provides device management, tolerance constants, and helper functions.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ==================== Device Configuration ====================


def get_available_devices() -> List[str]:
    """
    Get list of available devices for testing.

    Returns:
        List of device strings: ['cpu'] or ['cpu', 'cuda'] if CUDA available
    """
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def skip_if_no_cuda(test_func):
    """Decorator to skip test if CUDA not available."""

    def wrapper(*args, **kwargs):
        if not is_cuda_available():
            print(f"  ⊘ Skipping CUDA test (CUDA not available)")
            return None
        return test_func(*args, **kwargs)

    return wrapper


# ==================== Tolerance Constants ====================

# Forward pass tolerances
FP32_ATOL = 1e-6
FP32_RTOL = 1e-5

# Gradient check tolerances
GRAD_ATOL = 1e-4
GRAD_RTOL = 1e-3

# CPU-GPU consistency tolerances
CPU_GPU_ATOL = 1e-5
CPU_GPU_RTOL = 1e-4


# ==================== Test Data Generation ====================


def generate_random_input(
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    seed: Optional[int] = 42,
) -> torch.Tensor:
    """
    Generate random input tensor.

    Args:
        shape: Tensor shape
        dtype: Data type
        device: Device ('cpu' or 'cuda')
        seed: Random seed for reproducibility

    Returns:
        Random tensor
    """
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(shape, dtype=dtype, device=device)


def generate_binary_input(
    shape: Tuple[int, ...], device: str = "cpu", seed: Optional[int] = 42
) -> torch.Tensor:
    """
    Generate binary input tensor in [0, 1].

    Args:
        shape: Tensor shape
        device: Device ('cpu' or 'cuda')
        seed: Random seed

    Returns:
        Binary tensor in [0, 1]
    """
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randint(0, 2, shape, dtype=torch.float32, device=device)


def generate_uniform_input(
    shape: Tuple[int, ...], device: str = "cpu", seed: Optional[int] = 42
) -> torch.Tensor:
    """
    Generate uniform random input in [0, 1].

    Args:
        shape: Tensor shape
        device: Device ('cpu' or 'cuda')
        seed: Random seed

    Returns:
        Uniform tensor in [0, 1]
    """
    if seed is not None:
        torch.manual_seed(seed)
    return torch.rand(shape, dtype=torch.float32, device=device)


# ==================== Assertion Helpers ====================


def assert_shape_equal(actual: torch.Tensor, expected: Tuple[int, ...], msg: str = ""):
    """Assert tensor shape matches expected."""
    assert actual.shape == expected, f"Shape mismatch: {actual.shape} != {expected}. {msg}"


def assert_range(
    tensor: torch.Tensor, min_val: float, max_val: float, msg: str = "", rtol: float = 1e-5
):
    """
    Assert all tensor values are in range [min_val, max_val].

    Args:
        tensor: Tensor to check
        min_val: Minimum expected value
        max_val: Maximum expected value
        msg: Additional message
        rtol: Relative tolerance for range check
    """
    actual_min = tensor.min().item()
    actual_max = tensor.max().item()
    tolerance = rtol * (max_val - min_val)

    assert actual_min >= min_val - tolerance, f"Min value {actual_min} < {min_val}. {msg}"
    assert actual_max <= max_val + tolerance, f"Max value {actual_max} > {max_val}. {msg}"


def assert_gradients_exist(module: nn.Module, msg: str = ""):
    """
    Assert that all parameters have non-zero gradients.

    Args:
        module: Module to check
        msg: Additional message
    """
    has_grads = False
    for param in module.parameters():
        if param.grad is not None:
            has_grads = True
            assert param.grad.abs().sum().item() > 0, f"Found zero gradient for parameter. {msg}"
    assert has_grads, f"No gradients found in module. {msg}"


def assert_tensors_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = FP32_ATOL,
    rtol: float = FP32_RTOL,
    msg: str = "",
):
    """
    Assert two tensors are close within tolerance.

    Args:
        actual: Actual tensor
        expected: Expected tensor
        atol: Absolute tolerance
        rtol: Relative tolerance
        msg: Additional message
    """
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        max_diff = (actual - expected).abs().max().item()
        raise AssertionError(
            f"Tensors not close: max_diff={max_diff}, atol={atol}, rtol={rtol}. {msg}"
        )


# ==================== CPU-GPU Consistency ====================


def compare_cpu_gpu_forward(
    module: nn.Module,
    input_tensor: torch.Tensor,
    atol: float = CPU_GPU_ATOL,
    rtol: float = CPU_GPU_RTOL,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compare forward pass on CPU vs GPU.

    Args:
        module: Module to test
        input_tensor: Input tensor (will be moved to each device)
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        Tuple of (cpu_output, gpu_output)

    Raises:
        AssertionError: If outputs don't match
        RuntimeError: If CUDA not available
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for GPU comparison test")

    # CPU forward pass
    module_cpu = module.cpu()
    input_cpu = input_tensor.cpu()
    with torch.no_grad():
        output_cpu = module_cpu(input_cpu)

    # GPU forward pass
    module_gpu = module.cuda()
    input_gpu = input_tensor.cuda()
    with torch.no_grad():
        output_gpu = module_gpu(input_gpu)

    # Move GPU output back to CPU for comparison
    output_gpu_cpu = output_gpu.cpu()

    # Check consistency
    assert_tensors_close(
        output_cpu, output_gpu_cpu, atol=atol, rtol=rtol, msg="CPU and GPU outputs differ"
    )

    return output_cpu, output_gpu_cpu


# ==================== Gradient Checking ====================


def compute_numerical_gradient(
    module: nn.Module,
    input_tensor: torch.Tensor,
    output_index: Optional[int] = None,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Compute numerical gradient using finite differences.

    Args:
        module: Module to test
        input_tensor: Input tensor (requires grad)
        output_index: Index of output to compute gradient for (0 if None)
        eps: Step size for finite differences

    Returns:
        Numerical gradient tensor
    """
    module.eval()
    numerical_grad = torch.zeros_like(input_tensor)

    for i in range(input_tensor.numel()):
        # Compute f(x + eps)
        input_plus = input_tensor.clone()
        input_plus.view(-1)[i] += eps
        with torch.no_grad():
            output_plus = module(input_plus)
        if isinstance(output_plus, tuple):
            output_plus = output_plus[0]
        f_plus = output_plus.sum() if output_index is None else output_plus.view(-1)[output_index]

        # Compute f(x - eps)
        input_minus = input_tensor.clone()
        input_minus.view(-1)[i] -= eps
        with torch.no_grad():
            output_minus = module(input_minus)
        if isinstance(output_minus, tuple):
            output_minus = output_minus[0]
        f_minus = (
            output_minus.sum() if output_index is None else output_minus.view(-1)[output_index]
        )

        # Finite difference
        numerical_grad.view(-1)[i] = (f_plus - f_minus) / (2 * eps)

    return numerical_grad


def check_gradients(
    module: nn.Module, input_tensor: torch.Tensor, atol: float = GRAD_ATOL, rtol: float = GRAD_RTOL
) -> bool:
    """
    Check that gradients are computed correctly using numerical gradient checking.

    Args:
        module: Module to test
        input_tensor: Input tensor
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        True if gradients match, False otherwise
    """
    module.train()
    input_tensor = input_tensor.requires_grad_(True)

    # Compute analytical gradients
    output = module(input_tensor)
    if isinstance(output, tuple):
        output = output[0]
    loss = output.sum()
    loss.backward()

    analytical_grad = input_tensor.grad.clone()

    # Compute numerical gradients
    input_tensor.grad = None
    module.eval()
    numerical_grad = compute_numerical_gradient(module, input_tensor.detach())

    # Compare
    try:
        assert_tensors_close(
            analytical_grad,
            numerical_grad,
            atol=atol,
            rtol=rtol,
            msg="Analytical and numerical gradients differ",
        )
        return True
    except AssertionError:
        return False


# ==================== Component Discovery ====================


def get_all_registered_nodes() -> Dict[str, Any]:
    """Get all registered nodes from registry."""
    from difflut.registry import REGISTRY

    nodes = {}
    for name in REGISTRY.list_nodes():
        try:
            nodes[name] = REGISTRY.get_node(name)
        except Exception as e:
            print(f"Warning: Failed to load node '{name}': {e}")
    return nodes


def get_all_registered_layers() -> Dict[str, Any]:
    """Get all registered layers from registry."""
    from difflut.registry import REGISTRY

    layers = {}
    for name in REGISTRY.list_layers():
        try:
            layers[name] = REGISTRY.get_layer(name)
        except Exception as e:
            print(f"Warning: Failed to load layer '{name}': {e}")
    return layers


def get_all_registered_encoders() -> Dict[str, Any]:
    """Get all registered encoders from registry."""
    from difflut.registry import REGISTRY

    encoders = {}
    for name in REGISTRY.list_encoders():
        try:
            encoders[name] = REGISTRY.get_encoder(name)
        except Exception as e:
            print(f"Warning: Failed to load encoder '{name}': {e}")
    return encoders


def get_all_registered_initializers() -> Dict[str, Any]:
    """Get all registered initializers from registry."""
    from difflut.registry import REGISTRY

    initializers = {}
    for name in REGISTRY.list_initializers():
        try:
            initializers[name] = REGISTRY.get_initializer(name)
        except Exception as e:
            print(f"Warning: Failed to load initializer '{name}': {e}")
    return initializers


def get_all_registered_regularizers() -> Dict[str, Any]:
    """Get all registered regularizers from registry."""
    from difflut.registry import REGISTRY

    regularizers = {}
    for name in REGISTRY.list_regularizers():
        try:
            regularizers[name] = REGISTRY.get_regularizer(name)
        except Exception as e:
            print(f"Warning: Failed to load regularizer '{name}': {e}")
    return regularizers


# ==================== Module Instantiation Helpers ====================


def instantiate_node(
    node_class: type,
    input_dim: int = 4,
    output_dim: int = 1,
    layer_size: int = 32,  # Kept for backward compatibility but ignored
    **kwargs,
) -> nn.Module:
    """
    Instantiate a node with default parameters.

    NOTE: layer_size parameter is kept for backward compatibility but is no longer used.
    Each node is now an independent instance processing 2D tensors (batch_size, input_dim).

    Args:
        node_class: Node class to instantiate
        input_dim: Input dimension
        output_dim: Output dimension
        layer_size: DEPRECATED - kept for backward compatibility, ignored
        **kwargs: Additional arguments

    Returns:
        Node instance
    """
    # New architecture: nodes no longer accept layer_size
    return node_class(input_dim=input_dim, output_dim=output_dim, **kwargs)


def instantiate_layer(
    layer_class: type,
    input_size: int = 256,
    output_size: int = 128,
    node_type: Optional[type] = None,
    n: int = 4,
    **kwargs,
) -> nn.Module:
    """
    Instantiate a layer with default parameters.

    Args:
        layer_class: Layer class to instantiate
        input_size: Input size
        output_size: Output size (number of nodes)
        node_type: Node type to use
        n: Number of inputs per node
        **kwargs: Additional arguments

    Returns:
        Layer instance
    """
    if node_type is None:
        from difflut.nodes import LinearLUTNode

        node_type = LinearLUTNode

    from difflut.nodes.node_config import NodeConfig

    node_config = NodeConfig(input_dim=n, output_dim=1)

    return layer_class(
        input_size=input_size,
        output_size=output_size,
        node_type=node_type,
        node_kwargs=node_config,
        **kwargs,
    )


def instantiate_encoder(encoder_class: type, num_bits: int = 8, **kwargs) -> nn.Module:
    """
    Instantiate an encoder with default parameters.

    Args:
        encoder_class: Encoder class to instantiate
        num_bits: Number of bits for encoding
        **kwargs: Additional arguments

    Returns:
        Encoder instance
    """
    return encoder_class(num_bits=num_bits, **kwargs)


# ==================== Context Managers ====================


class IgnoreWarnings:
    """Context manager to temporarily ignore warnings."""

    def __init__(self, category=UserWarning):
        self.category = category

    def __enter__(self):
        self.original_filters = warnings.filters[:]
        warnings.filterwarnings("ignore", category=self.category)
        return self

    def __exit__(self, *args):
        warnings.filters[:] = self.original_filters


# ==================== Printing Helpers ====================


def print_section(title: str, width: int = 70):
    """Print a formatted section header."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_subsection(title: str, width: int = 70):
    """Print a formatted subsection header."""
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")


def print_test_result(test_name: str, passed: bool, message: str = ""):
    """Print formatted test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {test_name}")
    if message:
        print(f"         {message}")


if __name__ == "__main__":
    import sys

    # test_utils.py is just utilities, not a test file
    # It should exit with 0 (success)
    print("✓ Test utilities module loaded successfully")
    sys.exit(0)
