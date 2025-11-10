import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

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
DEFAULT_KAIMING_MODE: str = "fan_in"
# Default nonlinearity for Kaiming/He ('relu', 'leaky_relu', 'tanh', 'sigmoid')
DEFAULT_KAIMING_NONLINEARITY: str = "leaky_relu"
# Default target variance for variance-stabilized initialization
DEFAULT_VARIANCE_STABILIZED_V_TARGET: float = 1.0
# Default number of samples for regularization computations
DEFAULT_REGULARIZER_NUM_SAMPLES: int = 100
# Default small standard deviation for residual initialization noise
DEFAULT_RESIDUAL_SIGMA: float = 0.01
# Default logit scale for LUT-based residual initialization
DEFAULT_RESIDUAL_LOGIT_SCALE: float = 10.0


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
def normal_init(
    param: torch.Tensor,
    mean: float = DEFAULT_NORMAL_INIT_MEAN,
    std: float = DEFAULT_NORMAL_INIT_STD,
    **kwargs,
) -> None:
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
def uniform_init(
    param: torch.Tensor,
    a: float = DEFAULT_UNIFORM_INIT_A,
    b: float = DEFAULT_UNIFORM_INIT_B,
    **kwargs,
) -> None:
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
def kaiming_uniform_init(
    param: torch.Tensor,
    a: float = DEFAULT_KAIMING_A,
    mode: str = DEFAULT_KAIMING_MODE,
    nonlinearity: str = DEFAULT_KAIMING_NONLINEARITY,
    **kwargs,
) -> None:
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

        fan = fan_in if mode == "fan_in" else fan_out
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        param.uniform_(-bound, bound)


@register_initializer("kaiming_normal")
@register_initializer("he_normal")
@register_initializer("kaiming")
@register_initializer("he")
def kaiming_normal_init(
    param: torch.Tensor,
    a: float = DEFAULT_KAIMING_A,
    mode: str = DEFAULT_KAIMING_MODE,
    nonlinearity: str = DEFAULT_KAIMING_NONLINEARITY,
    **kwargs,
) -> None:
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

        fan = fan_in if mode == "fan_in" else fan_out
        std = gain / math.sqrt(fan)
        param.normal_(0, std)


@register_initializer("probabilistic_init")
@register_initializer("variance_stabilized")
def variance_stabilized_init(
    param: torch.Tensor,
    v_target: float = DEFAULT_VARIANCE_STABILIZED_V_TARGET,
    fan_in: Optional[int] = None,
    fan_out: Optional[int] = None,
    **kwargs,
) -> None:
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

    IMPORTANT: fan_in and fan_out refer to the LOGICAL node dimensions:
    - fan_in: Number of Boolean inputs to the node (k, the input_dim)
    - fan_out: Number of outputs from the node (m, the output_dim)

    NOT the truth table dimensions! For a probabilistic node with param shape
    (2^k, m), we have fan_in=k and fan_out=m (not 2^k and m).

    Args:
        param: The parameter tensor to initialize (e.g., raw_weights of shape (2^k, m))
        v_target: Target output variance (default: 1.0)
        fan_in: Number of Boolean inputs (k). If None, will infer from log2(table_size)
        fan_out: Number of outputs (m). If None, will infer as output_dim from shape
        **kwargs: Unused, for API consistency
    """
    if v_target == DEFAULT_VARIANCE_STABILIZED_V_TARGET:
        warn_default_value("v_target (variance_stabilized_init)", v_target, stacklevel=3)
    if fan_in is None:
        warn_default_value("fan_in (variance_stabilized_init)", "inferred from shape", stacklevel=3)
    if fan_out is None:
        warn_default_value(
            "fan_out (variance_stabilized_init)", "inferred from shape", stacklevel=3
        )

    # Infer fan_in and fan_out from parameter shape if not provided
    if fan_in is None or fan_out is None:
        if param.dim() >= 2:
            # For LUT nodes: shape is typically (2^input_dim, output_dim) or (num_table_entries, output_dim)
            # fan_in should be input_dim (log2 of table size)
            # fan_out should be output_dim (last dimension)
            table_size = param.shape[0]
            output_dim = param.shape[-1]

            if fan_in is None:
                # Try to infer input_dim from table size (assuming it's 2^k)
                log_size = math.log2(table_size)
                if abs(log_size - round(log_size)) < 1e-6:
                    # Table size is a power of 2, likely a truth table
                    fan_in = int(round(log_size))
                else:
                    # Not a power of 2, use table size as fallback
                    fan_in = table_size

            if fan_out is None:
                fan_out = output_dim
        else:
            # 1D parameter - use size for both
            fan_in = param.shape[0] if fan_in is None else fan_in
            fan_out = param.shape[0] if fan_out is None else fan_out

    # Calculate variance according to the forward variance consistency theorem
    # σ_c² = (v_target * 4) / (k + m)
    sigma_c_squared = (v_target * 4.0) / (fan_in + fan_out)
    std = math.sqrt(sigma_c_squared)

    # Initialize the parameter with this variance
    with torch.no_grad():
        param.normal_(0, std)


# ============================================================================
# Residual Initialization Strategies
# ============================================================================


def _residual_init_linear(
    param: torch.Tensor, sigma_small: float = DEFAULT_RESIDUAL_SIGMA
) -> None:
    """
    Linear residual initialization for linear nodes.

    Initializes weights to emphasize the first input component:
        c_i = 1 if i=1, else ε_i ~ N(0, σ²_small)

    This makes the node primarily sensitive to the first input while
    maintaining differentiability.

    Args:
        param: Weight vector of shape (input_dim, output_dim)
        sigma_small: Standard deviation for small noise terms
    """
    with torch.no_grad():
        # Initialize all weights with small random noise
        param.normal_(0, sigma_small)

        # Set first input weight to 1 for all outputs
        if param.dim() >= 2:
            param[0, :] = 1.0
        else:
            param[0] = 1.0


def _residual_init_polynomial(
    param: torch.Tensor,
    monomial_combinations: list,
    sigma_small: float = DEFAULT_RESIDUAL_SIGMA,
) -> None:
    """
    Polynomial residual initialization for polynomial nodes.

    Initializes coefficients to capture first-order dependencies on the first input:
        c_α = 1 if α=(1,0,...,0), else ε_α ~ N(0, σ²_small)

    This focuses the polynomial expansion on linear terms involving the first
    input component.

    Args:
        param: Coefficient tensor of shape (num_monomials, output_dim)
        monomial_combinations: List of monomial exponent tuples
        sigma_small: Standard deviation for small noise terms
    """
    with torch.no_grad():
        # Initialize all coefficients with small random noise
        param.normal_(0, sigma_small)

        # Find the index of the (1,0,...,0) monomial
        # This represents the linear term for the first input
        first_input_linear_idx = None
        for idx, exponents in enumerate(monomial_combinations):
            # Check if this is (1, 0, 0, ..., 0)
            if exponents[0] == 1 and all(e == 0 for e in exponents[1:]):
                first_input_linear_idx = idx
                break

        # Set coefficient for first input linear term to 1
        if first_input_linear_idx is not None:
            if param.dim() >= 2:
                param[first_input_linear_idx, :] = 1.0
            else:
                param[first_input_linear_idx] = 1.0


def _residual_init_mlp(
    param: torch.Tensor,
    param_name: str,
    layer_idx: int,
    num_layers: int,
    sigma_small: float = DEFAULT_RESIDUAL_SIGMA,
) -> None:
    """
    MLP residual initialization for neural network nodes.

    Employs skip connection-aware initialization:
        W^(1)_{i,j} = 1 if i=j=1, else ε_{i,j} ~ N(0, σ²_small)
        b^(l) = 0

    The final layer weights are initialized to pass through the first component
    of the transformed features, leveraging skip connections to preserve the
    first input signal.

    Args:
        param: Weight or bias tensor
        param_name: Name of the parameter ('weight' or 'bias')
        layer_idx: Index of the current layer (0-based)
        num_layers: Total number of layers
        sigma_small: Standard deviation for small noise terms
    """
    with torch.no_grad():
        if "bias" in param_name.lower():
            # All biases initialized to zero
            param.zero_()
        elif "weight" in param_name.lower():
            # Initialize weights with small random noise
            param.normal_(0, sigma_small)

            # For the first layer, set W[0,0] = 1 (pass through first input)
            if layer_idx == 0 and param.dim() >= 2:
                param[0, 0] = 1.0


def _residual_init_fourier(
    param: torch.Tensor,
    param_name: str,
    num_frequencies: int,
    sigma_small: float = DEFAULT_RESIDUAL_SIGMA,
) -> None:
    """
    Fourier residual initialization for Fourier nodes.

    Initializes to capture low-frequency components aligned with the first input:
        A_1 = 1, φ_1 = 0, b = 0.5, k_1 = (0.5, 0, ..., 0)

    Other frequencies are initialized with small random amplitudes and random phases.

    Args:
        param: Parameter tensor (amplitudes, phases, or bias)
        param_name: Name of the parameter ('amplitudes', 'phases', or 'bias')
        num_frequencies: Total number of frequency components
        sigma_small: Standard deviation for small amplitude noise
    """
    with torch.no_grad():
        if "amplitude" in param_name.lower():
            # Initialize amplitudes with small random values
            param.normal_(0, sigma_small)
            param.abs_()  # Amplitudes should be positive

            # Set first frequency amplitude to 1
            if param.dim() >= 2:
                param[0, :] = 1.0
            else:
                param[0] = 1.0

        elif "phase" in param_name.lower():
            # Initialize phases randomly in [-π, π]
            param.uniform_(-math.pi, math.pi)

            # Set first frequency phase to 0
            if param.dim() >= 2:
                param[0, :] = 0.0
            else:
                param[0] = 0.0

        elif "bias" in param_name.lower():
            # Initialize bias to 0.5 (center of [0,1] range)
            param.fill_(0.5)


def _residual_init_lut(
    param: torch.Tensor,
    input_dim: int,
    logit_scale: float = DEFAULT_RESIDUAL_LOGIT_SCALE,
) -> None:
    """
    LUT-based residual initialization for weightless, probabilistic, and hybrid nodes.

    Implements truth table initialization that passes through the first input:
        sigmoid(c)_{ι(a)} = 1 if a_1=1, else 0 for all a ∈ {0,1}^k

    This is achieved by setting raw logits to large positive values for patterns
    where a_1=1 and large negative values for patterns where a_1=0.

    Args:
        param: Raw logits tensor of shape (2^input_dim, output_dim) or (output_dim, 2^input_dim)
        input_dim: Number of Boolean inputs (k)
        logit_scale: Scale for the logit values (default: 10.0)
    """
    with torch.no_grad():
        # Determine the shape orientation
        if param.shape[0] == 2**input_dim:
            # Shape is (2^input_dim, output_dim) - typical for probabilistic nodes
            table_size, output_dim = param.shape
            transpose = False
        elif param.shape[1] == 2**input_dim:
            # Shape is (output_dim, 2^input_dim) - typical for DWN/hybrid nodes
            output_dim, table_size = param.shape
            transpose = True
        else:
            # Cannot determine proper orientation, initialize with small noise
            param.normal_(0, DEFAULT_RESIDUAL_SIGMA)
            return

        # Generate all binary combinations for input_dim bits
        # For each combination, check if the first bit (a_1) is 1
        for idx in range(table_size):
            # Extract bits in reverse order (LSB to MSB)
            # Bit 0 (LSB) corresponds to the last input, bit (input_dim-1) to the first
            bits = [(idx >> bit_pos) & 1 for bit_pos in range(input_dim)]

            # Check the first input (most significant bit in terms of logic)
            # In the indexing scheme, this is the last bit extracted
            first_input_value = bits[-1]

            # Set logit based on first input value
            if first_input_value == 1:
                logit_value = logit_scale  # High logit → sigmoid ≈ 1
            else:
                logit_value = -logit_scale  # Low logit → sigmoid ≈ 0

            # Set the value based on orientation
            if transpose:
                param[:, idx] = logit_value
            else:
                param[idx, :] = logit_value


@register_initializer("residual")
def residual_init(
    param: torch.Tensor,
    node_type: Optional[str] = None,
    param_name: Optional[str] = None,
    input_dim: Optional[int] = None,
    output_dim: Optional[int] = None,
    sigma_small: float = DEFAULT_RESIDUAL_SIGMA,
    logit_scale: float = DEFAULT_RESIDUAL_LOGIT_SCALE,
    # Polynomial-specific
    monomial_combinations: Optional[list] = None,
    # MLP-specific
    layer_idx: Optional[int] = None,
    num_layers: Optional[int] = None,
    # Fourier-specific
    num_frequencies: Optional[int] = None,
    **kwargs,
) -> None:
    """
    Residual initialization strategy for differentiable LUT nodes.

    Initializes nodes to approximate identity-like functions that pass through
    the first input, providing meaningful initial function representations while
    maintaining the ability to learn complex Boolean functions during training.

    The specific initialization strategy depends on the node type:

    **Linear Nodes** (`node_type='linear_lut'`):
        Initializes weight vector c ∈ R^k to emphasize the first input:
            c_i = 1 if i=1, else ε_i ~ N(0, σ²_small)

    **Polynomial Nodes** (`node_type='polylut'`):
        Initializes coefficients to capture first-order dependencies:
            c_α = 1 if α=(1,0,...,0), else ε_α ~ N(0, σ²_small)

    **MLP Nodes** (`node_type='neurallut'`):
        Skip connection-aware initialization:
            W^(1)_{i,j} = 1 if i=j=1, else ε_{i,j} ~ N(0, σ²_small)
            b^(l) = 0

    **Fourier Nodes** (`node_type='fourier'`):
        Low-frequency initialization aligned with first input:
            A_1 = 1, φ_1 = 0, b = 0.5, k_1 = (0.5, 0, ..., 0)
        Other frequencies: A_k ~ N(0, σ²_small), φ_k ~ Uniform(-π, π)

    **LUT-based Nodes** (`node_type` in ['dwn', 'probabilistic', 'hybrid']):
        Truth table initialization for identity on first input:
            sigmoid(c)_{ι(a)} = 1 if a_1=1, else 0, ∀a ∈ {0,1}^k

    Args:
        param: The parameter tensor to initialize
        node_type: Type of node ('linear_lut', 'polylut', 'neurallut', 'fourier',
                   'dwn', 'probabilistic', 'hybrid', etc.)
        param_name: Name of the parameter being initialized
        input_dim: Number of inputs (k) - required for LUT-based nodes
        output_dim: Number of outputs (m)
        sigma_small: Standard deviation for small random perturbations (default: 0.01)
        logit_scale: Scale for logit values in LUT-based initialization (default: 10.0)
        monomial_combinations: List of monomial exponent tuples (for polynomial nodes)
        layer_idx: Layer index for MLP nodes (0-based)
        num_layers: Total number of layers for MLP nodes
        num_frequencies: Number of frequency components for Fourier nodes
        **kwargs: Additional node-specific parameters

    Raises:
        ValueError: If required parameters are missing for the specified node type

    Examples:
        >>> # Linear node initialization
        >>> weights = torch.zeros(6, 1)  # 6 inputs, 1 output
        >>> residual_init(weights, node_type='linear_lut')

        >>> # Polynomial node initialization
        >>> coeffs = torch.zeros(num_monomials, 1)
        >>> residual_init(coeffs, node_type='polylut',
        ...              monomial_combinations=monomial_list)

        >>> # LUT-based node initialization
        >>> raw_weights = torch.zeros(64, 1)  # 2^6 entries, 1 output
        >>> residual_init(raw_weights, node_type='probabilistic', input_dim=6)
    """
    # Warn about default values
    if sigma_small == DEFAULT_RESIDUAL_SIGMA:
        warn_default_value("sigma_small (residual_init)", sigma_small, stacklevel=3)
    if logit_scale == DEFAULT_RESIDUAL_LOGIT_SCALE:
        warn_default_value("logit_scale (residual_init)", logit_scale, stacklevel=3)

    # Validate node_type is provided
    if node_type is None:
        raise ValueError(
            "residual_init requires 'node_type' to determine initialization strategy. "
            "Valid types: 'linear_lut', 'polylut', 'neurallut', 'fourier', "
            "'dwn', 'probabilistic', 'hybrid'"
        )

    # Dispatch to appropriate initialization strategy
    node_type_lower = node_type.lower()

    if node_type_lower == "linear_lut":
        _residual_init_linear(param, sigma_small=sigma_small)

    elif node_type_lower == "polylut":
        if monomial_combinations is None:
            raise ValueError("residual_init for polylut nodes requires 'monomial_combinations'")
        _residual_init_polynomial(
            param, monomial_combinations=monomial_combinations, sigma_small=sigma_small
        )

    elif node_type_lower == "neurallut":
        if layer_idx is None or num_layers is None:
            raise ValueError("residual_init for neurallut nodes requires 'layer_idx' and 'num_layers'")
        if param_name is None:
            raise ValueError("residual_init for neurallut nodes requires 'param_name'")
        _residual_init_mlp(
            param,
            param_name=param_name,
            layer_idx=layer_idx,
            num_layers=num_layers,
            sigma_small=sigma_small,
        )

    elif node_type_lower == "fourier":
        if num_frequencies is None:
            raise ValueError("residual_init for fourier nodes requires 'num_frequencies'")
        if param_name is None:
            raise ValueError("residual_init for fourier nodes requires 'param_name'")
        _residual_init_fourier(
            param, param_name=param_name, num_frequencies=num_frequencies, sigma_small=sigma_small
        )

    elif node_type_lower in ["dwn", "dwn_stable", "probabilistic", "hybrid"]:
        if input_dim is None:
            raise ValueError(
                f"residual_init for {node_type_lower} nodes requires 'input_dim' "
                "to determine truth table size"
            )
        _residual_init_lut(param, input_dim=input_dim, logit_scale=logit_scale)

    else:
        raise ValueError(
            f"Unknown node_type '{node_type}' for residual initialization. "
            "Valid types: 'linear_lut', 'polylut', 'neurallut', 'fourier', "
            "'dwn', 'dwn_stable', 'probabilistic', 'hybrid'"
        )

