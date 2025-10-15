"""
Common regularization functions for DiffLUT nodes.

Each regularizer is a function that takes a node as input and returns a scalar tensor.
These can be passed to nodes via the regularizers parameter.

Example usage:
    from difflut.utils.regularizers import l2_weights, l1_weights
    
    node = DWNNode(
        num_inputs=6,
        regularizers={
            "l2": [l2_weights, 0.01],
            "l1": [l1_weights, 0.001]
        }
    )
"""

import torch
import torch.nn as nn


def l2_weights(node: nn.Module) -> torch.Tensor:
    """
    L2 (weight decay) regularization on all parameters.
    
    Args:
        node: The node to regularize
        
    Returns:
        Sum of squared values of all parameters
    """
    reg = torch.tensor(0.0, device=next(node.parameters()).device)
    for param in node.parameters():
        reg = reg + torch.sum(param ** 2)
    return reg


def l1_weights(node: nn.Module) -> torch.Tensor:
    """
    L1 (lasso) regularization on all parameters.
    
    Args:
        node: The node to regularize
        
    Returns:
        Sum of absolute values of all parameters
    """
    reg = torch.tensor(0.0, device=next(node.parameters()).device)
    for param in node.parameters():
        reg = reg + torch.sum(torch.abs(param))
    return reg


def l2_lut_only(node: nn.Module) -> torch.Tensor:
    """
    L2 regularization specifically on LUT tables (parameters named 'luts').
    Useful for DWN nodes.
    
    Args:
        node: The node to regularize
        
    Returns:
        Sum of squared LUT values
    """
    reg = torch.tensor(0.0, device=next(node.parameters()).device)
    for name, param in node.named_parameters():
        if 'lut' in name.lower():
            reg = reg + torch.sum(param ** 2)
    return reg


def l1_lut_only(node: nn.Module) -> torch.Tensor:
    """
    L1 regularization specifically on LUT tables (parameters named 'luts').
    Useful for DWN nodes.
    
    Args:
        node: The node to regularize
        
    Returns:
        Sum of absolute LUT values
    """
    reg = torch.tensor(0.0, device=next(node.parameters()).device)
    for name, param in node.named_parameters():
        if 'lut' in name.lower():
            reg = reg + torch.sum(torch.abs(param))
    return reg


def entropy_regularizer(node: nn.Module) -> torch.Tensor:
    """
    Entropy regularization to encourage diversity in LUT entries.
    Useful for preventing mode collapse.
    
    Args:
        node: The node to regularize
        
    Returns:
        Negative entropy (lower is more diverse)
    """
    reg = torch.tensor(0.0, device=next(node.parameters()).device)
    
    for name, param in node.named_parameters():
        if 'lut' in name.lower() or 'weight' in name.lower():
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(param.flatten())
            # Clamp to avoid log(0)
            probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
            # Compute entropy
            entropy = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs))
            reg = reg - entropy.mean()  # Negative because we want to maximize entropy
    
    return reg


def sparsity_regularizer(node: nn.Module, threshold: float = 0.1) -> torch.Tensor:
    """
    Sparsity regularization to encourage parameters close to zero.
    
    Args:
        node: The node to regularize
        threshold: Values below this threshold are encouraged
        
    Returns:
        Number of parameters above threshold (normalized)
    """
    reg = torch.tensor(0.0, device=next(node.parameters()).device)
    total_params = 0
    
    for param in node.parameters():
        above_threshold = (torch.abs(param) > threshold).float()
        reg = reg + torch.sum(above_threshold)
        total_params += param.numel()
    
    # Normalize by total number of parameters
    if total_params > 0:
        reg = reg / total_params
    
    return reg


def gradient_penalty(node: nn.Module) -> torch.Tensor:
    """
    Gradient penalty to smooth the LUT function.
    Encourages neighboring LUT entries to have similar values.
    
    Args:
        node: The node to regularize
        
    Returns:
        Sum of squared differences between adjacent LUT entries
    """
    reg = torch.tensor(0.0, device=next(node.parameters()).device)
    
    for name, param in node.named_parameters():
        if 'lut' in name.lower() and param.dim() >= 2:
            # Compute differences between consecutive entries
            diff = param[:, 1:] - param[:, :-1]
            reg = reg + torch.sum(diff ** 2)
    
    return reg


def orthogonality_regularizer(node: nn.Module) -> torch.Tensor:
    """
    Orthogonality regularization for weight matrices.
    Encourages weight matrices to be orthogonal (useful for MLPs).
    
    Args:
        node: The node to regularize
        
    Returns:
        Deviation from orthogonality
    """
    reg = torch.tensor(0.0, device=next(node.parameters()).device)
    
    for name, param in node.named_parameters():
        if 'weight' in name.lower() and param.dim() == 2:
            # Compute W @ W^T
            prod = torch.matmul(param, param.t())
            # Should be close to identity
            identity = torch.eye(prod.shape[0], device=param.device)
            reg = reg + torch.sum((prod - identity) ** 2)
    
    return reg


# Convenient presets
COMMON_REGULARIZERS = {
    "l2_light": [l2_weights, 0.0001],
    "l2_medium": [l2_weights, 0.001],
    "l2_heavy": [l2_weights, 0.01],
    "l1_light": [l1_weights, 0.0001],
    "l1_medium": [l1_weights, 0.001],
    "l1_heavy": [l1_weights, 0.01],
    "lut_l2": [l2_lut_only, 0.001],
    "lut_l1": [l1_lut_only, 0.001],
    "entropy": [entropy_regularizer, 0.01],
    "sparsity": [sparsity_regularizer, 0.01],
    "smooth": [gradient_penalty, 0.001],
    "orthogonal": [orthogonality_regularizer, 0.001],
}
