

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any
from ...registry import register_initializer


@register_initializer("zeros")
def zeros_init(param: torch.Tensor, **kwargs) -> None:
    """
    Initialize a parameter to zeros.
    
    Args:
        param: The parameter tensor to initialize
        **kwargs: Unused, for API consistency
    """
    with torch.no_grad():
        param.zero_()


@register_initializer("ones")
def ones_init(param: torch.Tensor, **kwargs) -> None:
    """
    Initialize a parameter to ones.
    
    Args:
        param: The parameter tensor to initialize
        **kwargs: Unused, for API consistency
    """
    with torch.no_grad():
        param.fill_(1.0)


@register_initializer("normal")
def normal_init(param: torch.Tensor, mean: float = 0.0, std: float = 1.0, **kwargs) -> None:
    """
    Initialize a parameter from a normal distribution.
    
    Args:
        param: The parameter tensor to initialize
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
        **kwargs: Unused, for API consistency
    """
    with torch.no_grad():
        param.normal_(mean, std)


@register_initializer("uniform")
def uniform_init(param: torch.Tensor, a: float = -1.0, b: float = 1.0, **kwargs) -> None:
    """
    Initialize a parameter from a uniform distribution.
    
    Args:
        param: The parameter tensor to initialize
        a: Lower bound of the uniform distribution
        b: Upper bound of the uniform distribution
        **kwargs: Unused, for API consistency
    """
    with torch.no_grad():
        param.uniform_(a, b)


@register_initializer("xavier_uniform")
@register_initializer("glorot_uniform")
@register_initializer("xavier")
def xavier_uniform_init(param: torch.Tensor, gain: float = 1.0, **kwargs) -> None:
    """
    Initialize a parameter using Xavier/Glorot uniform initialization.
    
    Args:
        param: The parameter tensor to initialize
        gain: Scaling factor for the initialization
        **kwargs: Unused, for API consistency
    """
    with torch.no_grad():
        # Calculate fan_in and fan_out from parameter shape
        if param.dim() >= 2:
            fan_in = param.shape[1] if param.dim() > 1 else param.shape[0]
            fan_out = param.shape[0]
        else:
            fan_in = param.shape[0]
            fan_out = param.shape[0]
        
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        param.uniform_(-a, a)


@register_initializer("xavier_normal")
@register_initializer("glorot_normal")
@register_initializer("glorot")
def xavier_normal_init(param: torch.Tensor, gain: float = 1.0, **kwargs) -> None:
    """
    Initialize a parameter using Xavier/Glorot normal initialization.
    
    Args:
        param: The parameter tensor to initialize
        gain: Scaling factor for the initialization
        **kwargs: Unused, for API consistency
    """
    with torch.no_grad():
        # Calculate fan_in and fan_out from parameter shape
        if param.dim() >= 2:
            fan_in = param.shape[1] if param.dim() > 1 else param.shape[0]
            fan_out = param.shape[0]
        else:
            fan_in = param.shape[0]
            fan_out = param.shape[0]
        
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        param.normal_(0, std)


@register_initializer("kaiming_uniform")
@register_initializer("he_uniform")
def kaiming_uniform_init(param: torch.Tensor, a: float = 0, mode: str = 'fan_in', 
                         nonlinearity: str = 'leaky_relu', **kwargs) -> None:
    """
    Initialize a parameter using Kaiming/He uniform initialization.
    
    Args:
        param: The parameter tensor to initialize
        a: Negative slope of the rectifier used after this layer (only for 'leaky_relu')
        mode: Either 'fan_in' or 'fan_out'
        nonlinearity: Type of nonlinearity ('relu', 'leaky_relu', 'tanh', 'sigmoid')
        **kwargs: Unused, for API consistency
    """
    gain = nn.init.calculate_gain(nonlinearity, a)
    
    with torch.no_grad():
        # Calculate fan_in and fan_out from parameter shape
        if param.dim() >= 2:
            fan_in = param.shape[1] if param.dim() > 1 else param.shape[0]
            fan_out = param.shape[0]
        else:
            fan_in = param.shape[0]
            fan_out = param.shape[0]
        
        fan = fan_in if mode == 'fan_in' else fan_out
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        param.uniform_(-bound, bound)


@register_initializer("kaiming_normal")
@register_initializer("he_normal")
@register_initializer("kaiming")
@register_initializer("he")
def kaiming_normal_init(param: torch.Tensor, a: float = 0, mode: str = 'fan_in', 
                        nonlinearity: str = 'leaky_relu', **kwargs) -> None:
    """
    Initialize a parameter using Kaiming/He normal initialization.
    
    Args:
        param: The parameter tensor to initialize
        a: Negative slope of the rectifier used after this layer (only for 'leaky_relu')
        mode: Either 'fan_in' or 'fan_out'
        nonlinearity: Type of nonlinearity ('relu', 'leaky_relu', 'tanh', 'sigmoid')
        **kwargs: Unused, for API consistency
    """
    gain = nn.init.calculate_gain(nonlinearity, a)
    
    with torch.no_grad():
        # Calculate fan_in and fan_out from parameter shape
        if param.dim() >= 2:
            fan_in = param.shape[1] if param.dim() > 1 else param.shape[0]
            fan_out = param.shape[0]
        else:
            fan_in = param.shape[0]
            fan_out = param.shape[0]
        
        fan = fan_in if mode == 'fan_in' else fan_out
        std = gain / math.sqrt(fan)
        param.normal_(0, std)


@register_initializer("probabilistic_init")
@register_initializer("variance_stabilized")
def variance_stabilized_init(param: torch.Tensor, v_target: float = 1.0, 
                             fan_in: Optional[int] = None, 
                             fan_out: Optional[int] = None,
                             **kwargs) -> None:
    """
    Variance-stabilized initialization for probabilistic LUT parameters.
    
    This initialization ensures forward variance consistency across layers by
    scaling the raw logits such that node outputs remain in a healthy range
    of the sigmoid, preventing saturation at 0 or 1.
    
    Based on Theorem (Forward Variance Consistency):
    For a probabilistic node with k inputs and m outputs, initializing each logit as:
        c_i ~ N(0, σ_c²), where σ_c² = (v_target * 4) / (k + m)
    
    ensures that the expected output variance remains approximately v_target
    across layers at initialization.
    
    The fan_in and fan_out are typically calculated as:
    - fan_in: Number of inputs to the node (k)
    - fan_out: Number of downstream connections = next_layer_width * input_dim
              which represents how many nodes in the next layer connect to this node's output.
              If not provided, uses inferred values from parameter shape.
    
    Args:
        param: The parameter tensor to initialize
        v_target: Target output variance (default: 1.0)
        fan_in: Number of inputs (k). If None, will infer from parameter shape
        fan_out: Number of outputs (m). If None, will infer from parameter shape
        **kwargs: Unused, for API consistency
    """
    # Infer fan_in and fan_out from parameter shape if not provided
    if fan_in is None or fan_out is None:
        if param.dim() >= 2:
            fan_in = param.shape[1] if fan_in is None else fan_in
            fan_out = param.shape[0] if fan_out is None else fan_out
        else:
            fan_in = param.shape[0] if fan_in is None else fan_in
            fan_out = param.shape[0] if fan_out is None else fan_out
    
    # Calculate variance according to the forward variance consistency theorem
    # σ_c² = (v_target * 4) / (k + m)
    sigma_c_squared = (v_target * 4.0) / (fan_in + fan_out)
    std = math.sqrt(sigma_c_squared)
    
    # Initialize the parameter with this variance
    with torch.no_grad():
        param.normal_(0, std)


