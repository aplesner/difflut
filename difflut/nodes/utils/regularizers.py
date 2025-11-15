from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from ...registry import register_regularizer
from ...utils.warnings import warn_default_value

# ==================== Default Values ====================
# Module-level constants for all default parameters (Python 3.6+ compatible)

# Norm parameter for functional regularization
DEFAULT_REGULARIZER_P_NORM: int = 2
# Number of random samples for regularization
DEFAULT_REGULARIZER_NUM_SAMPLES: int = 100


# Helper function (not registered)
def _generate_hamming_neighbors(z: torch.Tensor) -> torch.Tensor:
    """
    Generate all Hamming neighbors of binary input z by flipping each bit.

    Args:
        z: Binary input tensor of shape (..., k) where k is the number of inputs

    Returns:
        Tensor of shape (..., k, k) where neighbors[..., i, :] is z with bit i flipped
    """
    k = z.shape[-1]
    # Expand z to shape (..., k, k)
    z_expanded = z.unsqueeze(-2).expand(*z.shape[:-1], k, k)
    # Create identity matrix to flip each bit
    flip_mask = torch.eye(k, device=z.device, dtype=z.dtype)
    # Flip bits: z^(i) has the i-th bit flipped
    neighbors = z_expanded.clone()
    neighbors = neighbors - 2 * z_expanded * flip_mask + flip_mask
    return neighbors


# Helper function (not registered)
def _l_regularizer_input_based(
    node: nn.Module, inputs: torch.Tensor, p: int
) -> torch.Tensor:
    """
    Input-based functional L-regularization for DiffLUT nodes.

    Measures sensitivity to bit flips on the *actual* batch inputs the model sees.
    This version is fully differentiable and data-aware.

    Args:
        node: The DiffLUT node to regularize
        inputs: Current batch inputs, shape (batch_size, k)
        p: The norm parameter (1 for L1, 2 for L2, or any positive value)

    Returns:
        Average functional sensitivity on current batch
    """
    k = inputs.shape[-1]
    batch_size = inputs.shape[0]

    # Generate Hamming neighbors for current inputs
    z_neighbors = _generate_hamming_neighbors(inputs)  # Shape: (batch_size, k, k)

    # Compute node output for original inputs (WITH gradients)
    g_z = node(inputs)  # Shape: (batch_size, output_dim)

    # Compute outputs for all neighbors (WITH gradients)
    z_neighbors_flat = z_neighbors.reshape(-1, k)
    # Shape: (batch_size * k, output_dim)
    g_z_neighbors_flat = node(z_neighbors_flat)
    g_z_neighbors = g_z_neighbors_flat.reshape(
        batch_size, k, -1
    )  # Shape: (batch_size, k, output_dim)

    # Compute differences |g(z) - g(z^(i))|^p for each neighbor i
    g_z_expanded = g_z.unsqueeze(1)  # Shape: (batch_size, 1, output_dim)
    # Shape: (batch_size, k, output_dim)
    differences = torch.abs(g_z_expanded - g_z_neighbors)

    # Apply p-norm
    if p == 1:
        sensitivity = differences
    elif p == 2:
        sensitivity = differences**2
    else:
        sensitivity = differences**p

    # Average over inputs (1/k factor) and sum over output dimensions
    reg = sensitivity.sum(dim=-1).mean(dim=-1).sum(dim=0) / k

    # Average over batch
    reg = reg / batch_size

    return reg


# Helper function (not registered)
def _l_regularizer_random(node: nn.Module, p: int, num_samples: int) -> torch.Tensor:
    """
    Random sampling functional L-regularization for DiffLUT nodes.

    Measures sensitivity to bit flips on random binary inputs.
    This version uses torch.no_grad() for efficiency but is not differentiable.

    Args:
        node: The DiffLUT node to regularize
        p: The norm parameter (1 for L1, 2 for L2, or any positive value)
        num_samples: Number of random binary input samples to evaluate

    Returns:
        Average functional sensitivity across sampled inputs
    """
    # Get device from node parameters
    device = next(node.parameters()).device

    # Infer number of inputs from node
    if hasattr(node, "num_inputs"):
        k = node.num_inputs
    elif hasattr(node, "in_features"):
        k = node.in_features
    elif hasattr(node, "luts"):
        # For DWN nodes with luts parameter
        luts = node.luts
        if luts.dim() == 2:
            k = int(torch.log2(torch.tensor(luts.shape[1], dtype=torch.float32)).item())
        else:
            raise ValueError("Cannot infer number of inputs from node")
    else:
        raise ValueError("Cannot infer number of inputs from node")

    # Sample random binary inputs from {0, 1}^k
    z = torch.randint(0, 2, (num_samples, k), device=device, dtype=torch.float32)

    # Generate Hamming neighbors: shape (num_samples, k, k)
    z_neighbors = _generate_hamming_neighbors(z)

    # Compute node output for original inputs
    with torch.no_grad():
        g_z = node(z)  # Shape: (num_samples, output_dim)

    # Compute node outputs for all neighbors
    # Reshape to (num_samples * k, k) for batch evaluation
    z_neighbors_flat = z_neighbors.reshape(-1, k)
    with torch.no_grad():
        # Shape: (num_samples * k, output_dim)
        g_z_neighbors_flat = node(z_neighbors_flat)
    g_z_neighbors = g_z_neighbors_flat.reshape(
        num_samples, k, -1
    )  # Shape: (num_samples, k, output_dim)

    # Compute differences |g(z) - g(z^(i))|^p for each neighbor i
    g_z_expanded = g_z.unsqueeze(1)  # Shape: (num_samples, 1, output_dim)
    # Shape: (num_samples, k, output_dim)
    differences = torch.abs(g_z_expanded - g_z_neighbors)

    # Apply p-norm
    if p == 1:
        sensitivity = differences
    elif p == 2:
        sensitivity = differences**2
    else:
        sensitivity = differences**p

    # Average over inputs (1/k factor) and sum over output dimensions
    reg = sensitivity.sum(dim=-1).mean(dim=-1).sum(dim=0) / k

    # Average over samples
    reg = reg / num_samples

    return reg


@register_regularizer("l")
@register_regularizer("functional")
def l_regularizer(
    node: nn.Module,
    p: int = DEFAULT_REGULARIZER_P_NORM,
    num_samples: int = DEFAULT_REGULARIZER_NUM_SAMPLES,
    inputs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Functional L-regularization for DiffLUT nodes.

    Parameters:
    - node: nn.Module, The DiffLUT node to regularize
    - p: int, Norm parameter (1 for L1, 2 for L2, default: 2)
    - num_samples: int, Random samples (only if inputs is None, default: 100)
    - inputs: Optional[torch.Tensor], Batch inputs for input-based mode

    Returns:
    - torch.Tensor, Average functional sensitivity
    """
    if p == DEFAULT_REGULARIZER_P_NORM:
        warn_default_value("p (l_regularizer)", p, stacklevel=3)

    # Dispatch to appropriate implementation
    if inputs is not None:
        # Input-based regularization (differentiable, data-aware)
        return _l_regularizer_input_based(node, inputs, p)
    else:
        # Random sampling regularization (non-differentiable, legacy)
        if num_samples == DEFAULT_REGULARIZER_NUM_SAMPLES:
            warn_default_value("num_samples (l_regularizer)", num_samples, stacklevel=3)
        return _l_regularizer_random(node, p, num_samples)


@register_regularizer("l1")
@register_regularizer("l1_functional")
def l1_regularizer(
    node: nn.Module,
    num_samples: int = DEFAULT_REGULARIZER_NUM_SAMPLES,
    inputs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    L1 functional regularization (wrapper for l_regularizer with p=1).

    Parameters:
    - node: nn.Module, The DiffLUT node to regularize
    - num_samples: int, Random samples (only if inputs is None, default: 100)
    - inputs: Optional[torch.Tensor], Batch inputs for input-based mode

    Returns:
    - torch.Tensor, Average L1 functional sensitivity
    """
    return l_regularizer(node, p=1, num_samples=num_samples, inputs=inputs)


@register_regularizer("l2")
@register_regularizer("l2_functional")
def l2_regularizer(
    node: nn.Module,
    num_samples: int = DEFAULT_REGULARIZER_NUM_SAMPLES,
    inputs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    L2 functional regularization (wrapper for l_regularizer with p=2).

    Parameters:
    - node: nn.Module, The DiffLUT node to regularize
    - num_samples: int, Random samples (only if inputs is None, default: 100)
    - inputs: Optional[torch.Tensor], Batch inputs for input-based mode

    Returns:
    - torch.Tensor, Average L2 functional sensitivity
    """
    return l_regularizer(node, p=2, num_samples=num_samples, inputs=inputs)


# Helper function (not registered)
def _compute_walsh_hadamard_matrix(k: int, device: torch.device) -> torch.Tensor:
    """
    Compute the Walsh-Hadamard coefficient matrix C for a k-input LUT.

    The matrix C has shape (2^k, 2^k) where:
        C[S, j] = (1 / 2^k) * prod_{i in S} (2 * j_i - 1)

    where j_i is the i-th bit in the binary representation of index j.

    Args:
        k: Number of LUT inputs

    Returns:
        Walsh-Hadamard coefficient matrix of shape (2^k, 2^k)
    """
    n = 2**k

    # Generate all binary indices j in {0, 1, ..., 2^k - 1}
    indices = torch.arange(n, device=device)

    # Extract binary representation: shape (n, k)
    binary_repr = torch.zeros((n, k), device=device)
    for i in range(k):
        binary_repr[:, i] = (indices >> i) & 1

    # Convert to {-1, 1} representation: 2 * j_i - 1
    binary_signs = 2 * binary_repr - 1  # Shape: (n, k)

    # Compute coefficient matrix
    # For each subset S (represented as row), compute product over elements in S
    C = torch.zeros((n, n), device=device)

    for s_idx in range(n):
        # Determine which bits are in subset S
        s_binary = torch.zeros(k, device=device)
        for i in range(k):
            s_binary[i] = (s_idx >> i) & 1

        # Compute product for each j
        # Product over i in S of (2 * j_i - 1)
        product = torch.ones(n, device=device)
        for i in range(k):
            if s_binary[i] == 1:
                product = product * binary_signs[:, i]

        C[s_idx, :] = product / n

    return C


@register_regularizer("spectral")
@register_regularizer("fourier")
@register_regularizer("walsh")
def spectral_regularizer(node: nn.Module) -> torch.Tensor:
    """
    Spectral regularization for truth-table parameterized DiffLUT nodes.

    Parameters:
    - node: nn.Module, The DiffLUT node to regularize

    Returns:
    - torch.Tensor, Spectral norm of the LUT function(s)
    """
    device = next(node.parameters()).device
    reg = None  # Start with None instead of 0.0 tensor

    # Look for truth-table parameters (typically named 'luts' or similar)
    for name, param in node.named_parameters():
        if "lut" in name.lower():
            # param should have shape (output_dim, 2^input_dim) for independent nodes
            # or (2^input_dim,) for a single-output node
            if param.dim() == 1:
                # Single output: shape (2^input_dim,)
                lut_table = param.unsqueeze(0)
            elif param.dim() == 2:
                # Multiple outputs: shape (output_dim, 2^input_dim)
                # Each row is an independent truth table for one output dimension
                lut_table = param
            else:
                # Skip if not a truth table
                continue

            # Infer k from table size
            table_size = lut_table.shape[1]
            k = int(torch.log2(torch.tensor(table_size, dtype=torch.float32)).item())

            # Verify that table_size is a power of 2
            if 2**k != table_size:
                continue  # Skip if not a valid truth table size

            # Compute Walsh-Hadamard coefficient matrix
            C = _compute_walsh_hadamard_matrix(k, device)

            # Compute spectral norm: ||L · C||_F^2
            # where L is the LUT table matrix (num_luts, 2^k)
            fourier_coeffs = torch.matmul(lut_table, C.T)  # Shape: (num_luts, 2^k)
            spectral_norm = torch.sum(fourier_coeffs**2)

            # Average over number of LUTs
            norm_contribution = spectral_norm / lut_table.shape[0]

            # Accumulate (preserves gradient)
            if reg is None:
                reg = norm_contribution
            else:
                reg = reg + norm_contribution

    # If no LUT parameters found, return zero tensor
    if reg is None:
        reg = torch.tensor(0.0, device=device, requires_grad=True)

    return reg
