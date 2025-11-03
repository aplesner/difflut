"""
Global constants for the difflut library.

This module centralizes all hand-tuned hyperparameters, thresholds, and default values
to improve maintainability and make tuning easier.
"""

# ============================================================================
# Layer Configuration and Validation Thresholds
# ============================================================================

# Threshold for warning about excessive input feature reuse in layers
# If total_connections > input_size * LAYER_REUSE_WARNING_THRESHOLD, warn user
LAYER_REUSE_WARNING_THRESHOLD = 100

# Divisor for warning about underutilization of input features in layers
# If output_size * n < input_size // LAYER_UNDERUSE_WARNING_DIVISOR, warn user
LAYER_UNDERUSE_WARNING_DIVISOR = 10

# Maximum recommended node input dimension before warning about memory usage
# If n > LAYER_MAX_NODE_INPUT_DIM, warn about exponential memory growth (2^n)
LAYER_MAX_NODE_INPUT_DIM = 15


# ============================================================================
# Node Default Values
# ============================================================================

# Default number of inputs per node if not specified
DEFAULT_NODE_INPUT_DIM = 6

# Default number of outputs per node if not specified
DEFAULT_NODE_OUTPUT_DIM = 1

# Default layer size (number of parallel nodes) if not specified
DEFAULT_NODE_LAYER_SIZE = 1

# Threshold for warning about large input dimensions
# If input_dim > NODE_INPUT_DIM_WARNING_THRESHOLD, warn about memory
NODE_INPUT_DIM_WARNING_THRESHOLD = 10

# Threshold for warning about large output dimensions
# If output_dim > NODE_OUTPUT_DIM_WARNING_THRESHOLD, warn about memory
NODE_OUTPUT_DIM_WARNING_THRESHOLD = 10


# ============================================================================
# DWN (Differentiable Weightless Network) Hyperparameters
# ============================================================================

# Base gradient scaling factor for DWN nodes
# Default alpha = DWN_ALPHA_BASE * (DWN_ALPHA_DECAY ** (n-1))
DWN_ALPHA_BASE = 0.5

# Decay factor for alpha based on number of inputs
# Applied as: alpha = 0.5 * (0.75 ** (n-1))
DWN_ALPHA_DECAY = 0.75

# Hamming distance decay factor for DWN backward pass
# Default beta = DWN_BETA_NUMERATOR / DWN_BETA_DENOMINATOR
DWN_BETA_NUMERATOR = 0.25
DWN_BETA_DENOMINATOR = 0.75

# Binary threshold for DWN forward pass
# Inputs >= DWN_BINARY_THRESHOLD are treated as 1, otherwise 0
DWN_BINARY_THRESHOLD = 0.5


# ============================================================================
# Encoder Default Values
# ============================================================================

# Default number of bits for encoding continuous values
DEFAULT_ENCODER_NUM_BITS = 3

# Default flatten behavior for encoders
# If True, output shape is (batch_size, input_dim * num_bits)
# If False, output shape is (batch_size, input_dim, num_bits)
DEFAULT_ENCODER_FLATTEN = True


# ============================================================================
# Thermometer Encoder Specific
# ============================================================================

# Default number of bits for thermometer encoding
DEFAULT_THERMOMETER_NUM_BITS = 1


# ============================================================================
# Advanced Encoder Specific
# ============================================================================

# Default number of bits for advanced encoders (Binary, Gray, etc.)
DEFAULT_ADVANCED_ENCODER_NUM_BITS = 8

# Default base for logarithmic encoder
DEFAULT_LOGARITHMIC_BASE = 2.0

# Minimum number of bits required for sign-magnitude encoding
MIN_SIGN_MAGNITUDE_BITS = 2
