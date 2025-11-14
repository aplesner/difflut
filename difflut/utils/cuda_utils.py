"""
CUDA utility functions for device-aware operations.

Provides standardized way to detect device and decide whether to use CUDA kernels.
This replaces the scattered use_cuda parameters across the codebase.

Key principle: Use PyTorch's standard `.to(device)` convention throughout.
CUDA kernels are automatically selected based on tensor/module device, not config.
"""

import torch
import torch.nn as nn
from typing import Union


def get_device_from_module(module: nn.Module) -> torch.device:
    """
    Get the device that a module is on.
    
    Args:
        module: A PyTorch module
        
    Returns:
        The device the module is on (cuda:0, cpu, etc.)
    """
    try:
        # Try to get device from first parameter
        return next(module.parameters()).device
    except StopIteration:
        # Module has no parameters, default to CPU
        return torch.device("cpu")


def get_device_from_tensor(tensor: torch.Tensor) -> torch.device:
    """
    Get the device that a tensor is on.
    
    Args:
        tensor: A PyTorch tensor
        
    Returns:
        The device the tensor is on
    """
    return tensor.device


def should_use_cuda(device: Union[torch.device, str, int, None]) -> bool:
    """
    Determine if CUDA kernels should be used based on device.
    
    This is the single source of truth for CUDA kernel selection throughout
    the package. Replaces all scattered use_cuda parameters.
    
    Args:
        device: Device to check (torch.device, string like "cuda", "cuda:0", "cpu", etc.)
        
    Returns:
        True if device is CUDA and CUDA kernels are available, False otherwise
    """
    if device is None:
        return False
        
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    
    # CUDA kernels should be used only if:
    # 1. The device is CUDA
    # 2. CUDA is available
    return device.type == "cuda" and torch.cuda.is_available()


def should_use_cuda_from_tensor(tensor: torch.Tensor) -> bool:
    """
    Convenience function: should we use CUDA kernels for this tensor?
    
    Args:
        tensor: Input tensor
        
    Returns:
        True if tensor is on CUDA device, False otherwise
    """
    return should_use_cuda(tensor.device)


def should_use_cuda_from_module(module: nn.Module) -> bool:
    """
    Convenience function: should we use CUDA kernels for this module?
    
    Args:
        module: PyTorch module
        
    Returns:
        True if module is on CUDA device, False otherwise
    """
    return should_use_cuda(get_device_from_module(module))


def ensure_same_device(module: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure a tensor is on the same device as a module.
    
    Args:
        module: PyTorch module (defines target device)
        tensor: Tensor to move
        
    Returns:
        Tensor on the same device as the module
    """
    module_device = get_device_from_module(module)
    return tensor.to(module_device)


def ensure_same_device_as_tensor(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor2 is on the same device as tensor1.
    
    Args:
        tensor1: Reference tensor (defines target device)
        tensor2: Tensor to move
        
    Returns:
        tensor2 on the same device as tensor1
    """
    return tensor2.to(tensor1.device)
