"""
CUDA kernels for DiffLUT nodes.

This module provides a simple utility function to check CUDA availability.
Node-specific CUDA extension imports and autograd functions are now in their respective node files:
- dwn_node.py: EFDFunction, efd_forward
- hybrid_node.py: HybridFunction, hybrid_forward
- fourier_node.py: FourierFunction, fourier_forward
"""

import torch


def is_cuda_available() -> bool:
    """Check if CUDA is available for use."""
    return torch.cuda.is_available()


__all__ = ["is_cuda_available"]
