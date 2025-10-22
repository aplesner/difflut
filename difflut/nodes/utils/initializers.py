"""
Weight initialization functions for DiffLUT nodes.

Each initializer is a function that takes a node and optional parameters,
and initializes the node's weights/parameters.

Example usage:
    from difflut.nodes.utils import get_initializer
    
    init_fn = get_initializer("variance_stabilized")
    node = ProbabilisticNode(
        num_inputs=6,
        init_fn=init_fn
    )
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any
from ...registry import register_initializer


@register_initializer("zeros")
def zeros_init(node: nn.Module, **kwargs) -> None:
    """
    Initialize all learnable parameters to zeros.
    
    Args:
        node: The DiffLUT node to initialize
        **kwargs: Unused, for API consistency
    """
    with torch.no_grad():
        for param in node.parameters():
            if param.requires_grad:
                param.zero_()


@register_initializer("ones")
def ones_init(node: nn.Module, **kwargs) -> None:
    """
    Initialize all learnable parameters to ones.
    
    Args:
        node: The DiffLUT node to initialize
        **kwargs: Unused, for API consistency
    """
    with torch.no_grad():
        for param in node.parameters():
            if param.requires_grad:
                param.fill_(1.0)


@register_initializer("normal")
@register_initializer("gaussian")
def normal_init(node: nn.Module, mean: float = 0.0, std: float = 1.0, **kwargs) -> None:
    """
    Initialize all learnable parameters from a normal/Gaussian distribution.
    
    Args:
        node: The DiffLUT node to initialize
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
        **kwargs: Unused, for API consistency
    """
    with torch.no_grad():
        for param in node.parameters():
            if param.requires_grad:
                param.normal_(mean, std)


@register_initializer("uniform")
def uniform_init(node: nn.Module, a: float = -1.0, b: float = 1.0, **kwargs) -> None:
    """
    Initialize all learnable parameters from a uniform distribution.
    
    Args:
        node: The DiffLUT node to initialize
        a: Lower bound of the uniform distribution
        b: Upper bound of the uniform distribution
        **kwargs: Unused, for API consistency
    """
    with torch.no_grad():
        for param in node.parameters():
            if param.requires_grad:
                param.uniform_(a, b)


@register_initializer("xavier_uniform")
@register_initializer("glorot_uniform")
@register_initializer("xavier")
def xavier_uniform_init(node: nn.Module, gain: float = 1.0, **kwargs) -> None:
    """
    Initialize parameters using Xavier/Glorot uniform initialization.
    
    Args:
        node: The DiffLUT node to initialize
        gain: Scaling factor for the initialization
        **kwargs: May contain 'fan_in' and 'fan_out' for custom fan calculations
    """
    fan_in = kwargs.get('fan_in', None)
    fan_out = kwargs.get('fan_out', None)
    
    with torch.no_grad():
        for param in node.parameters():
            if param.requires_grad:
                # Calculate fan_in and fan_out from parameter shape if not provided
                if fan_in is None or fan_out is None:
                    if param.dim() >= 2:
                        param_fan_in = param.shape[1] if param.dim() > 1 else param.shape[0]
                        param_fan_out = param.shape[0]
                    else:
                        param_fan_in = param.shape[0]
                        param_fan_out = param.shape[0]
                else:
                    param_fan_in = fan_in
                    param_fan_out = fan_out
                
                std = gain * math.sqrt(2.0 / (param_fan_in + param_fan_out))
                a = math.sqrt(3.0) * std
                param.uniform_(-a, a)


@register_initializer("xavier_normal")
@register_initializer("glorot_normal")
@register_initializer("glorot")
def xavier_normal_init(node: nn.Module, gain: float = 1.0, **kwargs) -> None:
    """
    Initialize parameters using Xavier/Glorot normal initialization.
    
    Args:
        node: The DiffLUT node to initialize
        gain: Scaling factor for the initialization
        **kwargs: May contain 'fan_in' and 'fan_out' for custom fan calculations
    """
    fan_in = kwargs.get('fan_in', None)
    fan_out = kwargs.get('fan_out', None)
    
    with torch.no_grad():
        for param in node.parameters():
            if param.requires_grad:
                # Calculate fan_in and fan_out from parameter shape if not provided
                if fan_in is None or fan_out is None:
                    if param.dim() >= 2:
                        param_fan_in = param.shape[1] if param.dim() > 1 else param.shape[0]
                        param_fan_out = param.shape[0]
                    else:
                        param_fan_in = param.shape[0]
                        param_fan_out = param.shape[0]
                else:
                    param_fan_in = fan_in
                    param_fan_out = fan_out
                
                std = gain * math.sqrt(2.0 / (param_fan_in + param_fan_out))
                param.normal_(0, std)


@register_initializer("kaiming_uniform")
@register_initializer("he_uniform")
def kaiming_uniform_init(node: nn.Module, a: float = 0, mode: str = 'fan_in', 
                         nonlinearity: str = 'leaky_relu', **kwargs) -> None:
    """
    Initialize parameters using Kaiming/He uniform initialization.
    
    Args:
        node: The DiffLUT node to initialize
        a: Negative slope of the rectifier used after this layer (only for 'leaky_relu')
        mode: Either 'fan_in' or 'fan_out'
        nonlinearity: Type of nonlinearity ('relu', 'leaky_relu', 'tanh', 'sigmoid')
        **kwargs: May contain 'fan_in' and 'fan_out' for custom fan calculations
    """
    fan_in = kwargs.get('fan_in', None)
    fan_out = kwargs.get('fan_out', None)
    
    gain = nn.init.calculate_gain(nonlinearity, a)
    
    with torch.no_grad():
        for param in node.parameters():
            if param.requires_grad:
                # Calculate fan_in and fan_out from parameter shape if not provided
                if fan_in is None or fan_out is None:
                    if param.dim() >= 2:
                        param_fan_in = param.shape[1] if param.dim() > 1 else param.shape[0]
                        param_fan_out = param.shape[0]
                    else:
                        param_fan_in = param.shape[0]
                        param_fan_out = param.shape[0]
                else:
                    param_fan_in = fan_in
                    param_fan_out = fan_out
                
                fan = param_fan_in if mode == 'fan_in' else param_fan_out
                std = gain / math.sqrt(fan)
                bound = math.sqrt(3.0) * std
                param.uniform_(-bound, bound)


@register_initializer("kaiming_normal")
@register_initializer("he_normal")
@register_initializer("kaiming")
@register_initializer("he")
def kaiming_normal_init(node: nn.Module, a: float = 0, mode: str = 'fan_in', 
                        nonlinearity: str = 'leaky_relu', **kwargs) -> None:
    """
    Initialize parameters using Kaiming/He normal initialization.
    
    Args:
        node: The DiffLUT node to initialize
        a: Negative slope of the rectifier used after this layer (only for 'leaky_relu')
        mode: Either 'fan_in' or 'fan_out'
        nonlinearity: Type of nonlinearity ('relu', 'leaky_relu', 'tanh', 'sigmoid')
        **kwargs: May contain 'fan_in' and 'fan_out' for custom fan calculations
    """
    fan_in = kwargs.get('fan_in', None)
    fan_out = kwargs.get('fan_out', None)
    
    gain = nn.init.calculate_gain(nonlinearity, a)
    
    with torch.no_grad():
        for param in node.parameters():
            if param.requires_grad:
                # Calculate fan_in and fan_out from parameter shape if not provided
                if fan_in is None or fan_out is None:
                    if param.dim() >= 2:
                        param_fan_in = param.shape[1] if param.dim() > 1 else param.shape[0]
                        param_fan_out = param.shape[0]
                    else:
                        param_fan_in = param.shape[0]
                        param_fan_out = param.shape[0]
                else:
                    param_fan_in = fan_in
                    param_fan_out = fan_out
                
                fan = param_fan_in if mode == 'fan_in' else param_fan_out
                std = gain / math.sqrt(fan)
                param.normal_(0, std)


@register_initializer("probabilistic_init")
@register_initializer("variance_stabilized")
def variance_stabilized_init(node: nn.Module, v_target: float = 1.0, 
                             fan_in: Optional[int] = None, 
                             fan_out: Optional[int] = None,
                             **kwargs) -> None:
    """
    Variance-stabilized initialization for probabilistic LUT nodes.
    
    This initialization ensures forward variance consistency across layers by
    scaling the raw logits such that node outputs remain in a healthy range
    of the sigmoid, preventing saturation at 0 or 1.
    
    Based on Theorem (Forward Variance Consistency):
    For a probabilistic node with k inputs and m outputs, initializing each logit as:
        c_i ~ N(0, σ_c²), where σ_c² = (v_target * 4) / (k + m)
    
    ensures that the expected output variance remains approximately v_target
    across layers at initialization.
    
    Args:
        node: The DiffLUT node to initialize (typically a probabilistic node)
        v_target: Target output variance (default: 1.0)
        fan_in: Number of inputs (k). If None, will try to infer from node
        fan_out: Number of outputs (m). If None, will try to infer from node
        **kwargs: Unused, for API consistency
        
    Raises:
        ValueError: If fan_in or fan_out cannot be determined
    """
    # Try to infer fan_in (k) and fan_out (m) from the node
    if fan_in is None:
        if hasattr(node, 'num_inputs'):
            fan_in = node.num_inputs
        elif hasattr(node, 'k'):
            fan_in = node.k
        elif hasattr(node, 'in_features'):
            fan_in = node.in_features
        else:
            raise ValueError(
                "Cannot infer fan_in (number of inputs k) from node. "
                "Please provide fan_in explicitly."
            )
    
    if fan_out is None:
        if hasattr(node, 'num_outputs'):
            fan_out = node.num_outputs
        elif hasattr(node, 'm'):
            fan_out = node.m
        elif hasattr(node, 'out_features'):
            fan_out = node.out_features
        elif hasattr(node, 'output_dim'):
            fan_out = node.output_dim
        else:
            # Default to 1 for single-output nodes
            fan_out = 1
    
    # Calculate variance according to the forward variance consistency theorem
    # σ_c² = (v_target * 4) / (k + m)
    sigma_c_squared = (v_target * 4.0) / (fan_in + fan_out)
    std = math.sqrt(sigma_c_squared)
    
    # Initialize all learnable parameters with this variance
    with torch.no_grad():
        for param in node.parameters():
            if param.requires_grad:
                param.normal_(0, std)


