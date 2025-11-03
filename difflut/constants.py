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
# Layer Default Values
# ============================================================================

# Default bit-flip probability for training augmentation
# If flip_probability > 0, randomly flip this fraction of bits during training
DEFAULT_LAYER_FLIP_PROBABILITY = 0.0

# Default gradient stabilization mode
# Options: 'none', 'layerwise', 'batchwise'
DEFAULT_LAYER_GRAD_STABILIZATION = 'none'

# Default target standard deviation for gradient stabilization
# Used when grad_stabilization is not 'none'
DEFAULT_LAYER_GRAD_TARGET_STD = 1.0

# Default flag for whether to subtract mean during gradient stabilization
DEFAULT_LAYER_GRAD_SUBTRACT_MEAN = False

# Default epsilon for numerical stability in gradient stabilization
DEFAULT_LAYER_GRAD_EPSILON = 1e-8

# Maximum number of nodes to process in a single batch during forward/backward pass
# Memory optimization: prevents OOM by processing large layers in chunks
# Trade-off: Smaller values = less memory, slightly more kernel launches
# Recommended values: 256 (memory-constrained), 512 (balanced), 1024 (high-memory GPUs)
# Set to -1 to disable batching (process all nodes at once)
DEFAULT_LAYER_MAX_NODES_PER_BATCH = 512


# ============================================================================
# Random Layer Specific Defaults
# ============================================================================

# Default random seed for reproducible random mapping
DEFAULT_RANDOM_LAYER_SEED = 42


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
# NeuralLUT Node Specific Defaults
# ============================================================================

# Default width of hidden layers in NeuralLUT MLP
DEFAULT_NEURALLUT_HIDDEN_WIDTH = 8

# Default number of layers in NeuralLUT MLP
DEFAULT_NEURALLUT_DEPTH = 2

# Default interval for skip connections in NeuralLUT (0 = no skips)
DEFAULT_NEURALLUT_SKIP_INTERVAL = 2

# Default activation function for NeuralLUT ('relu' or 'sigmoid')
DEFAULT_NEURALLUT_ACTIVATION = 'relu'

# Default starting temperature for NeuralLUT
DEFAULT_NEURALLUT_TAU_START = 1.0

# Default minimum temperature for NeuralLUT
DEFAULT_NEURALLUT_TAU_MIN = 0.0001

# Default temperature decay iterations for NeuralLUT
DEFAULT_NEURALLUT_TAU_DECAY_ITERS = 1000.0

# Default flag for using Straight-Through Estimator in NeuralLUT
DEFAULT_NEURALLUT_STE = False

# Default gradient scaling factor for NeuralLUT
DEFAULT_NEURALLUT_GRAD_FACTOR = 1.0


# ============================================================================
# Learnable Layer Specific Defaults
# ============================================================================

# Default temperature for softmax in learnable mapping
DEFAULT_LEARNABLE_LAYER_TAU = 0.001

# Default starting value for tau (used for exponential decay)
DEFAULT_LEARNABLE_LAYER_TAU_START = 1.0

# Default minimum value tau can decay to
DEFAULT_LEARNABLE_LAYER_TAU_MIN = 0.0001

# Default number of iterations for tau to decay by factor of 10
DEFAULT_LEARNABLE_LAYER_TAU_DECAY_ITERS = 1000.0

# Threshold for warning about excessive learnable connections
# If output_size * n > input_size * LEARNABLE_LAYER_CONNECTION_WARNING_THRESHOLD, warn
LEARNABLE_LAYER_CONNECTION_WARNING_THRESHOLD = 10

# Use CUDA kernel for soft selection (training mode)
# If False, always use PyTorch's matmul (more stable, well-tested)
# If True, use custom CUDA kernel (potentially faster for large matrices)
# Note: Hard selection (eval mode) always uses CUDA kernel when available
DEFAULT_LEARNABLE_LAYER_USE_CUDA_SOFT = False


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
