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
# Default noise factor for residual initialization (0 = perfect residual, higher = more noise)
DEFAULT_RESIDUAL_NOISE_FACTOR: float = 0.0
# Default logit clarity for LUT-based residual initialization (higher = clearer separation)
DEFAULT_RESIDUAL_LOGIT_CLARITY: float = 5.0


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
    param: torch.Tensor,
    noise_factor: float = DEFAULT_RESIDUAL_NOISE_FACTOR,
    logit_clarity: float = DEFAULT_RESIDUAL_LOGIT_CLARITY,
    param_name: Optional[str] = None,
) -> None:
    """
    Linear residual initialization for linear nodes.

    Initializes weights and bias to pass through the first input:
        For weights:
            c_i = 2*logit_clarity if i=0, else ε_i ~ N(0, noise_factor²)
        For bias:
            b = -logit_clarity

    This creates a shifted sigmoid that passes through first input:
        - When x[0]=0: z = -logit_clarity, sigmoid(-5) ≈ 0.007
        - When x[0]=1: z = -logit_clarity + 2*logit_clarity = +logit_clarity, sigmoid(5) ≈ 0.993

    When noise_factor=0, this gives perfect pass-through during forward_eval.
    During forward_train with sigmoid, logit_clarity=5 gives sigmoid(±5)≈0.993/0.007.

    Args:
        param: Weight vector of shape (input_dim, output_dim) or bias vector of shape (output_dim,)
        noise_factor: Standard deviation for noise on other inputs (0 = no noise)
        logit_clarity: Scale for first input weight (default: 5.0)
        param_name: Name of parameter ('weights' or 'bias')
    """
    with torch.no_grad():
        if param_name == "bias":
            # Initialize bias to -logit_clarity (offset for proper sigmoid behavior)
            param.fill_(-logit_clarity)
        else:
            # Initialize weights
            if noise_factor > 0:
                # Initialize all weights with noise
                param.normal_(0, noise_factor)
            else:
                # Perfect initialization - all zeros
                param.zero_()

            # Set first input weight to 2*logit_clarity for all outputs
            if param.dim() >= 2:
                param[0, :] = 2 * logit_clarity
            else:
                param[0] = 2 * logit_clarity


def _residual_init_polynomial(
    param: torch.Tensor,
    monomial_combinations: list,
    noise_factor: float = DEFAULT_RESIDUAL_NOISE_FACTOR,
    logit_clarity: float = DEFAULT_RESIDUAL_LOGIT_CLARITY,
) -> None:
    """
    Polynomial residual initialization for polynomial nodes.

    Initializes coefficients to pass through the first input:
        c_α = 2*logit_clarity if α=(1,0,...,0) [first input linear term]
        c_α = -logit_clarity if α=(0,0,...,0) [constant/bias term]
        c_α = ε_α ~ N(0, noise_factor²) otherwise

    This creates a shifted sigmoid that passes through first input:
        - When x[0]=0: z = -logit_clarity, sigmoid(-5) ≈ 0.007
        - When x[0]=1: z = -logit_clarity + 2*logit_clarity = +logit_clarity, sigmoid(5) ≈ 0.993

    When noise_factor=0, this gives perfect pass-through during forward_eval.

    Args:
        param: Coefficient tensor of shape (num_monomials, output_dim)
        monomial_combinations: List of monomial exponent tuples
        noise_factor: Standard deviation for noise on other coefficients (0 = no noise)
        logit_clarity: Scale for first input linear term (default: 5.0)
    """
    with torch.no_grad():
        if noise_factor > 0:
            # Initialize all coefficients with noise
            param.normal_(0, noise_factor)
        else:
            # Perfect initialization - all zeros
            param.zero_()

        # Find the indices of the constant and first input linear terms
        constant_idx = None
        first_input_linear_idx = None

        for idx, exponents in enumerate(monomial_combinations):
            # Check if this is the constant term (0, 0, 0, ..., 0)
            if all(e == 0 for e in exponents):
                constant_idx = idx
            # Check if this is the first input linear term (1, 0, 0, ..., 0)
            elif exponents[0] == 1 and all(e == 0 for e in exponents[1:]):
                first_input_linear_idx = idx

        # Set coefficient for constant term to -logit_clarity (bias offset)
        if constant_idx is not None:
            if param.dim() >= 2:
                param[constant_idx, :] = -logit_clarity
            else:
                param[constant_idx] = -logit_clarity

        # Set coefficient for first input linear term to 2*logit_clarity
        if first_input_linear_idx is not None:
            if param.dim() >= 2:
                param[first_input_linear_idx, :] = 2 * logit_clarity
            else:
                param[first_input_linear_idx] = 2 * logit_clarity


def _residual_init_mlp(
    param: torch.Tensor,
    param_name: str,
    layer_idx: int,
    num_layers: int,
    noise_factor: float = DEFAULT_RESIDUAL_NOISE_FACTOR,
    logit_clarity: float = DEFAULT_RESIDUAL_LOGIT_CLARITY,
) -> None:
    """
    MLP residual initialization for neural network nodes.

    Employs skip connection-aware initialization:
        W^(1)_{i,j} = logit_clarity if i=j=0, else ε_{i,j} ~ N(0, noise_factor²)
        b^(l) = 0

    The final layer weights are initialized to pass through the first component
    of the transformed features, leveraging skip connections to preserve the
    first input signal.

    Args:
        param: Weight or bias tensor
        param_name: Name of the parameter ('weight' or 'bias')
        layer_idx: Index of the current layer (0-based)
        num_layers: Total number of layers
        noise_factor: Standard deviation for noise on other weights (0 = no noise)
        logit_clarity: Scale for diagonal/pass-through weights (default: 5.0)
    """
    with torch.no_grad():
        if "bias" in param_name.lower():
            # All biases initialized to zero
            param.zero_()
        elif "weight" in param_name.lower():
            if noise_factor > 0:
                # Initialize weights with noise
                param.normal_(0, noise_factor)
            else:
                # Perfect initialization - all zeros
                param.zero_()

            # For the first layer, set W[0,0] = logit_clarity (pass through first input)
            if layer_idx == 0 and param.dim() >= 2:
                param[0, 0] = logit_clarity


def _residual_init_fourier(
    param: torch.Tensor,
    param_name: str,
    num_frequencies: int,
    noise_factor: float = DEFAULT_RESIDUAL_NOISE_FACTOR,
    logit_clarity: float = DEFAULT_RESIDUAL_LOGIT_CLARITY,
) -> None:
    """
    Fourier residual initialization for Fourier nodes.

    Initializes to capture low-frequency components aligned with the first input:
        A_1 = logit_clarity, φ_1 = 0, b = 0.5, k_1 = (0.5, 0, ..., 0)

    Other frequencies are initialized with small random amplitudes and random phases.

    Args:
        param: Parameter tensor (amplitudes, phases, or bias)
        param_name: Name of the parameter ('amplitudes', 'phases', or 'bias')
        num_frequencies: Total number of frequency components
        noise_factor: Standard deviation for amplitude noise on non-primary frequencies
        logit_clarity: Amplitude for primary frequency (default: 5.0)
    """
    with torch.no_grad():
        if "amplitude" in param_name.lower():
            if noise_factor > 0:
                # Initialize amplitudes with noise
                param.normal_(0, noise_factor)
                param.abs_()  # Amplitudes should be positive
            else:
                # Perfect initialization - all zeros
                param.zero_()

            # Set first frequency amplitude to logit_clarity
            if param.dim() >= 2:
                param[0, :] = logit_clarity
            else:
                param[0] = logit_clarity

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
    noise_factor: float = DEFAULT_RESIDUAL_NOISE_FACTOR,
    logit_clarity: float = DEFAULT_RESIDUAL_LOGIT_CLARITY,
) -> None:
    """
    LUT-based residual initialization for weightless, probabilistic, and hybrid nodes.

    Implements truth table initialization that passes through the first input:
        For noise_factor=0:
            raw_lut[idx] = +logit_clarity if first_bit(idx)=1, else -logit_clarity
        For noise_factor>0:
            raw_lut[idx] = base_value + N(0, noise_factor²)

    This ensures that after sigmoid:
        - When first input = 1: output ≈ 1 (sigmoid(+logit_clarity) ≈ 0.993 for clarity=5)
        - When first input = 0: output ≈ 0 (sigmoid(-logit_clarity) ≈ 0.007 for clarity=5)
        - For noise_factor=0: Perfect residual initialization
        - For noise_factor>0: Residual initialization with Gaussian noise added

    Args:
        param: Raw logits tensor of shape (2^input_dim, output_dim) or (output_dim, 2^input_dim)
        input_dim: Number of Boolean inputs (k)
        noise_factor: Standard deviation for Gaussian noise (0 = perfect residual)
        logit_clarity: Scale for the logit values (default: 5.0, giving sigmoid ≈ 0.993/0.007)
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
            # Cannot determine proper orientation, initialize with noise
            if noise_factor > 0:
                param.normal_(0, noise_factor)
            else:
                param.zero_()
            return

        # Generate all binary combinations for input_dim bits
        # For each combination, check if the first bit (first input) is 1
        for idx in range(table_size):
            # Extract bits: bit i = (idx >> i) & 1
            # Bit ordering: idx = b_0 + b_1*2 + b_2*4 + ... + b_{n-1}*2^{n-1}
            # The first input corresponds to bit 0 (LSB), i.e., b_0
            first_input_bit = idx & 1  # Extract bit 0 (LSB)

            # Set logit based on first input bit
            if first_input_bit == 1:
                base_value = logit_clarity  # High logit → sigmoid ≈ 1
            else:
                base_value = -logit_clarity  # Low logit → sigmoid ≈ 0

            # Add noise if requested
            if noise_factor > 0:
                if transpose:
                    noise = (
                        torch.randn(output_dim, device=param.device, dtype=param.dtype)
                        * noise_factor
                    )
                    param[:, idx] = base_value + noise
                else:
                    noise = (
                        torch.randn(output_dim, device=param.device, dtype=param.dtype)
                        * noise_factor
                    )
                    param[idx, :] = base_value + noise
            else:
                # Perfect initialization - no noise
                if transpose:
                    param[:, idx] = base_value
                else:
                    param[idx, :] = base_value


@register_initializer("residual")
def residual_init(
    param: torch.Tensor,
    node_type: Optional[str] = None,
    param_name: Optional[str] = None,
    input_dim: Optional[int] = None,
    output_dim: Optional[int] = None,
    noise_factor: float = DEFAULT_RESIDUAL_NOISE_FACTOR,
    logit_clarity: float = DEFAULT_RESIDUAL_LOGIT_CLARITY,
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

    Initializes nodes to pass through the first input exactly (when noise_factor=0)
    or approximately (when noise_factor>0). This provides a meaningful starting point
    where output = first_input, allowing the network to learn from identity.

    The specific initialization strategy depends on the node type:

    **Linear Nodes** (`node_type='linear_lut'`):
        Initializes weight vector to pass through first input:
            c_0 = 2*logit_clarity, c_i = N(0, noise_factor²) for i>0

    **Polynomial Nodes** (`node_type='polylut'`):
        Initializes coefficients to pass through first input:
            c_{(1,0,...,0)} = 2*logit_clarity, others = N(0, noise_factor²)

    **MLP Nodes** (`node_type='neurallut'`):
        Skip connection-aware initialization:
            W^(1)_{0,0} = logit_clarity, others = N(0, noise_factor²)
            All biases = 0

    **Fourier Nodes** (`node_type='fourier'`):
        Low-frequency initialization aligned with first input:
            A_0 = logit_clarity, φ_0 = 0, bias = 0.5
            Other amplitudes = N(0, noise_factor²), phases = Uniform(-π, π)

    **LUT-based Nodes** (`node_type` in ['dwn', 'dwn_stable', 'probabilistic', 'hybrid']):
        Truth table initialization for first input pass-through:
            raw_lut[idx] = ±logit_clarity (depending on first bit) + N(0, noise_factor²)

        For noise_factor=0 and logit_clarity=5:
            - forward_eval: Perfect pass-through (output = first_input)
            - forward_train: Near-perfect (sigmoid(±5) ≈ 0.993/0.007)

    Args:
        param: The parameter tensor to initialize
        node_type: Type of node ('linear_lut', 'polylut', 'neurallut', 'fourier',
                   'dwn', 'dwn_stable', 'probabilistic', 'hybrid', etc.)
        param_name: Name of the parameter being initialized
        input_dim: Number of inputs - required for LUT-based nodes
        output_dim: Number of outputs
        noise_factor: Std dev for Gaussian noise (0 = perfect residual, default: 0.0)
        logit_clarity: Scale for pass-through weights/logits (default: 5.0)
        monomial_combinations: List of monomial exponent tuples (for polynomial nodes)
        layer_idx: Layer index for MLP nodes (0-based)
        num_layers: Total number of layers for MLP nodes
        num_frequencies: Number of frequency components for Fourier nodes
        **kwargs: Additional node-specific parameters

    Raises:
        ValueError: If required parameters are missing for the specified node type

    Examples:
        >>> # Perfect residual initialization (no noise)
        >>> residual_init(weights, node_type='linear_lut', noise_factor=0.0)

        >>> # Residual with small noise
        >>> residual_init(weights, node_type='dwn', input_dim=6, noise_factor=0.01)

        >>> # Custom logit clarity for stronger signal
        >>> residual_init(luts, node_type='hybrid', input_dim=6, logit_clarity=10.0)
    """
    # Warn about default values
    if noise_factor == DEFAULT_RESIDUAL_NOISE_FACTOR:
        warn_default_value("noise_factor (residual_init)", noise_factor, stacklevel=3)
    if logit_clarity == DEFAULT_RESIDUAL_LOGIT_CLARITY:
        warn_default_value("logit_clarity (residual_init)", logit_clarity, stacklevel=3)

    # Validate node_type is provided
    if node_type is None:
        raise ValueError(
            "residual_init requires 'node_type' to determine initialization strategy. "
            "Valid types: 'linear_lut', 'polylut', 'neurallut', 'fourier', "
            "'dwn', 'dwn_stable', 'probabilistic', 'hybrid'"
        )

    # Dispatch to appropriate initialization strategy
    node_type_lower = node_type.lower()

    if node_type_lower == "linear_lut":
        _residual_init_linear(
            param,
            noise_factor=noise_factor,
            logit_clarity=logit_clarity,
            param_name=param_name,
        )

    elif node_type_lower == "polylut":
        if monomial_combinations is None:
            raise ValueError("residual_init for polylut nodes requires 'monomial_combinations'")
        _residual_init_polynomial(
            param,
            monomial_combinations=monomial_combinations,
            noise_factor=noise_factor,
            logit_clarity=logit_clarity,
        )

    elif node_type_lower == "neurallut":
        if layer_idx is None or num_layers is None:
            raise ValueError(
                "residual_init for neurallut nodes requires 'layer_idx' and 'num_layers'"
            )
        if param_name is None:
            raise ValueError("residual_init for neurallut nodes requires 'param_name'")
        _residual_init_mlp(
            param,
            param_name=param_name,
            layer_idx=layer_idx,
            num_layers=num_layers,
            noise_factor=noise_factor,
            logit_clarity=logit_clarity,
        )

    elif node_type_lower == "fourier":
        if num_frequencies is None:
            raise ValueError("residual_init for fourier nodes requires 'num_frequencies'")
        if param_name is None:
            raise ValueError("residual_init for fourier nodes requires 'param_name'")
        _residual_init_fourier(
            param,
            param_name=param_name,
            num_frequencies=num_frequencies,
            noise_factor=noise_factor,
            logit_clarity=logit_clarity,
        )

    elif node_type_lower in ["dwn", "dwn_stable", "probabilistic", "hybrid"]:
        if input_dim is None:
            raise ValueError(
                f"residual_init for {node_type_lower} nodes requires 'input_dim' "
                "to determine truth table size"
            )
        _residual_init_lut(
            param,
            input_dim=input_dim,
            noise_factor=noise_factor,
            logit_clarity=logit_clarity,
        )

    else:
        raise ValueError(
            f"Unknown node_type '{node_type}' for residual initialization. "
            "Valid types: 'linear_lut', 'polylut', 'neurallut', 'fourier', "
            "'dwn', 'dwn_stable', 'probabilistic', 'hybrid'"
        )
