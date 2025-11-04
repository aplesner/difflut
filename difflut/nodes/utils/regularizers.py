import torch
import torch.nn as nn
from typing import Optional
from ...registry import register_regularizer
from ...utils.warnings import warn_default_value

# Default parameters for regularizers

# Default norm parameter for functional regularization (1 for L1, 2 for L2)
DEFAULT_REGULARIZER_P_NORM: int = 2
# Default number of random binary samples for regularization
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


@register_regularizer("l")
@register_regularizer("functional")
def l_regularizer(node: nn.Module, p: int = DEFAULT_REGULARIZER_P_NORM, num_samples: int = DEFAULT_REGULARIZER_NUM_SAMPLES) -> torch.Tensor:
    """
    Functional L-regularization for DiffLUT nodes.
    
    Measures how sensitive a node's output is to single-bit flips in its inputs.
    This is the continuous relaxation of Boolean sensitivity that encourages
    smooth and robust node behavior.
    
    The regularization is defined as:
        R_L(g; z) = (1/k) * sum_i |g(z) - g(z^(i))|^p
    
    where z^(i) is z with the i-th bit flipped, and k is the number of inputs.
    
    Args:
        node: The DiffLUT node to regularize
        p: The norm parameter (1 for L1, 2 for L2, or any positive value)
        num_samples: Number of random binary input samples to evaluate
        
    Returns:
        Average functional sensitivity across sampled inputs
    """
    if p == DEFAULT_REGULARIZER_P_NORM:
        warn_default_value("p (l_regularizer)", p, stacklevel=3)
    if num_samples == DEFAULT_REGULARIZER_NUM_SAMPLES:
        warn_default_value("num_samples (l_regularizer)", num_samples, stacklevel=3)
    # Get device from node parameters
    device = next(node.parameters()).device
    
    # Infer number of inputs from node
    if hasattr(node, 'num_inputs'):
        k = node.num_inputs
    elif hasattr(node, 'in_features'):
        k = node.in_features
    elif hasattr(node, 'luts'):
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
        g_z_neighbors_flat = node(z_neighbors_flat)  # Shape: (num_samples * k, output_dim)
    g_z_neighbors = g_z_neighbors_flat.reshape(num_samples, k, -1)  # Shape: (num_samples, k, output_dim)
    
    # Compute differences |g(z) - g(z^(i))|^p for each neighbor i
    g_z_expanded = g_z.unsqueeze(1)  # Shape: (num_samples, 1, output_dim)
    differences = torch.abs(g_z_expanded - g_z_neighbors)  # Shape: (num_samples, k, output_dim)
    
    # Apply p-norm
    if p == 1:
        sensitivity = differences
    elif p == 2:
        sensitivity = differences ** 2
    else:
        sensitivity = differences ** p
    
    # Average over inputs (1/k factor) and sum over output dimensions
    reg = sensitivity.sum(dim=-1).mean(dim=-1).sum(dim=0) / k
    
    # Average over samples
    reg = reg / num_samples
    
    return reg


@register_regularizer("l1")
@register_regularizer("l1_functional")
def l1_regularizer(node: nn.Module, num_samples: int = DEFAULT_REGULARIZER_NUM_SAMPLES) -> torch.Tensor:
    """
    L1 functional regularization (convenience wrapper for l_regularizer with p=1).
    
    Args:
        node: The DiffLUT node to regularize
        num_samples: Number of random binary input samples to evaluate
        
    Returns:
        Average L1 functional sensitivity
    """
    return l_regularizer(node, p=1, num_samples=num_samples)


@register_regularizer("l2")
@register_regularizer("l2_functional")
def l2_regularizer(node: nn.Module, num_samples: int = DEFAULT_REGULARIZER_NUM_SAMPLES) -> torch.Tensor:
    """
    L2 functional regularization (convenience wrapper for l_regularizer with p=2).
    
    Args:
        node: The DiffLUT node to regularize
        num_samples: Number of random binary input samples to evaluate
        
    Returns:
        Average L2 functional sensitivity
    """
    return l_regularizer(node, p=2, num_samples=num_samples)


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
    n = 2 ** k
    
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
    
    Penalizes the Fourier spectrum (Walsh-Hadamard transform) of the LUT function,
    encouraging low-frequency Boolean functions that are smooth and have low
    complexity. This is particularly effective for DWN nodes.
    
    The regularization is defined as:
        R_spec(g) = ||C · c||_2^2
    
    where c is the truth table vector and C is the Walsh-Hadamard coefficient matrix.
    
    Args:
        node: The DiffLUT node to regularize (must have truth-table parameters)
        
    Returns:
        Spectral norm of the LUT function(s)
    """
    device = next(node.parameters()).device
    reg = torch.tensor(0.0, device=device)
    
    # Look for truth-table parameters (typically named 'luts' or similar)
    for name, param in node.named_parameters():
        if 'lut' in name.lower():
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
            if 2 ** k != table_size:
                continue  # Skip if not a valid truth table size
            
            # Compute Walsh-Hadamard coefficient matrix
            C = _compute_walsh_hadamard_matrix(k, device)
            
            # Compute spectral norm: ||L · C||_F^2
            # where L is the LUT table matrix (num_luts, 2^k)
            fourier_coeffs = torch.matmul(lut_table, C.T)  # Shape: (num_luts, 2^k)
            spectral_norm = torch.sum(fourier_coeffs ** 2)
            
            # Average over number of LUTs
            reg = reg + spectral_norm / lut_table.shape[0]
    
    return reg




