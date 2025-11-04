import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any
from ...registry import register_initializer
from ...utils.warnings import warn_default_value

# Default parameters for initializers

# Default mean for normal initialization
DEFAULT_NORMAL_INIT_MEAN: float = 0.0
# Default standard deviation for normal initialization
DEFAULT_NORMAL_INIT_STD: float = 1.0
# Default lower bound for uniform initialization
DEFAULT_UNIFORM_INIT_A: float = -1.0
# Default upper bound for uniform initialization
DEFAULT_UNIFORM_INIT_B: float = 1.0
# Default gain factor for Xavier/Glorot initialization
DEFAULT_XAVIER_GAIN: float = 1.0
# Default gain factor for Kaiming/He initialization
DEFAULT_KAIMING_GAIN: float = 1.0
# Default negative slope for Kaiming/He (leaky_relu parameter)
DEFAULT_KAIMING_A: float = 0.0
# Default mode for Kaiming/He initialization ('fan_in' or 'fan_out')
DEFAULT_KAIMING_MODE: str = 'fan_in'
# Default nonlinearity for Kaiming/He ('relu', 'leaky_relu', 'tanh', 'sigmoid')
DEFAULT_KAIMING_NONLINEARITY: str = 'leaky_relu'
# Default target variance for variance-stabilized initialization
DEFAULT_VARIANCE_STABILIZED_V_TARGET: float = 1.0
# Default number of samples for regularization computations
DEFAULT_REGULARIZER_NUM_SAMPLES: int = 100


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
def normal_init(param: torch.Tensor, mean: float = DEFAULT_NORMAL_INIT_MEAN, std: float = DEFAULT_NORMAL_INIT_STD, **kwargs) -> None:
    """
    Initialize a parameter from a normal distribution.
    
    Args:
        param: The parameter tensor to initialize
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
        **kwargs: Unused, for API consistency
    """
    if mean == DEFAULT_NORMAL_INIT_MEAN:
        warn_default_value("mean (normal_init)", mean, stacklevel=3)
    if std == DEFAULT_NORMAL_INIT_STD:
        warn_default_value("std (normal_init)", std, stacklevel=3)
    
    with torch.no_grad():
        param.normal_(mean, std)


@register_initializer("uniform")
def uniform_init(param: torch.Tensor, a: float = DEFAULT_UNIFORM_INIT_A, b: float = DEFAULT_UNIFORM_INIT_B, **kwargs) -> None:
    """
    Initialize a parameter from a uniform distribution.
    
    Args:
        param: The parameter tensor to initialize
        a: Lower bound of the uniform distribution
        b: Upper bound of the uniform distribution
        **kwargs: Unused, for API consistency
    """
    if a == DEFAULT_UNIFORM_INIT_A:
        warn_default_value("a (uniform_init)", a, stacklevel=3)
    if b == DEFAULT_UNIFORM_INIT_B:
        warn_default_value("b (uniform_init)", b, stacklevel=3)
    
    with torch.no_grad():
        param.uniform_(a, b)


@register_initializer("xavier_uniform")
@register_initializer("glorot_uniform")
@register_initializer("xavier")
def xavier_uniform_init(param: torch.Tensor, gain: float = DEFAULT_XAVIER_GAIN, **kwargs) -> None:
    """
    Initialize a parameter using Xavier/Glorot uniform initialization.
    
    Args:
        param: The parameter tensor to initialize
        gain: Scaling factor for the initialization
        **kwargs: Unused, for API consistency
    """
    if gain == DEFAULT_XAVIER_GAIN:
        warn_default_value("gain (xavier_uniform_init)", gain, stacklevel=3)
    
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
def xavier_normal_init(param: torch.Tensor, gain: float = DEFAULT_XAVIER_GAIN, **kwargs) -> None:
    """
    Initialize a parameter using Xavier/Glorot normal initialization.
    
    Args:
        param: The parameter tensor to initialize
        gain: Scaling factor for the initialization
        **kwargs: Unused, for API consistency
    """
    if gain == DEFAULT_XAVIER_GAIN:
        warn_default_value("gain (xavier_normal_init)", gain, stacklevel=3)
    
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
def kaiming_uniform_init(param: torch.Tensor, a: float = DEFAULT_KAIMING_A, mode: str = DEFAULT_KAIMING_MODE, 
                         nonlinearity: str = DEFAULT_KAIMING_NONLINEARITY, **kwargs) -> None:
    """
    Initialize a parameter using Kaiming/He uniform initialization.
    
    Args:
        param: The parameter tensor to initialize
        a: Negative slope of the rectifier used after this layer (only for 'leaky_relu')
        mode: Either 'fan_in' or 'fan_out'
        nonlinearity: Type of nonlinearity ('relu', 'leaky_relu', 'tanh', 'sigmoid')
        **kwargs: Unused, for API consistency
    """
    if a == DEFAULT_KAIMING_A:
        warn_default_value("a (kaiming_uniform_init)", a, stacklevel=3)
    if mode == DEFAULT_KAIMING_MODE:
        warn_default_value("mode (kaiming_uniform_init)", mode, stacklevel=3)
    if nonlinearity == DEFAULT_KAIMING_NONLINEARITY:
        warn_default_value("nonlinearity (kaiming_uniform_init)", nonlinearity, stacklevel=3)
    
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
def kaiming_normal_init(param: torch.Tensor, a: float = DEFAULT_KAIMING_A, mode: str = DEFAULT_KAIMING_MODE, 
                        nonlinearity: str = DEFAULT_KAIMING_NONLINEARITY, **kwargs) -> None:
    """
    Initialize a parameter using Kaiming/He normal initialization.
    
    Args:
        param: The parameter tensor to initialize
        a: Negative slope of the rectifier used after this layer (only for 'leaky_relu')
        mode: Either 'fan_in' or 'fan_out'
        nonlinearity: Type of nonlinearity ('relu', 'leaky_relu', 'tanh', 'sigmoid')
        **kwargs: Unused, for API consistency
    """
    if a == DEFAULT_KAIMING_A:
        warn_default_value("a (kaiming_normal_init)", a, stacklevel=3)
    if mode == DEFAULT_KAIMING_MODE:
        warn_default_value("mode (kaiming_normal_init)", mode, stacklevel=3)
    if nonlinearity == DEFAULT_KAIMING_NONLINEARITY:
        warn_default_value("nonlinearity (kaiming_normal_init)", nonlinearity, stacklevel=3)
    
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
def variance_stabilized_init(param: torch.Tensor, v_target: float = DEFAULT_VARIANCE_STABILIZED_V_TARGET, 
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
    if v_target == DEFAULT_VARIANCE_STABILIZED_V_TARGET:
        warn_default_value("v_target (variance_stabilized_init)", v_target, stacklevel=3)
    if fan_in is None:
        warn_default_value("fan_in (variance_stabilized_init)", "inferred from shape", stacklevel=3)
    if fan_out is None:
        warn_default_value("fan_out (variance_stabilized_init)", "inferred from shape", stacklevel=3)
    
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


