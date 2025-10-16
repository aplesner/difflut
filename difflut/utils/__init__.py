"""DiffLUT utility functions and modules."""

from .regularizers import (
    l1_weights,
    l2_weights,
    l1_lut_only,
    l2_lut_only,
    entropy_regularizer,
    sparsity_regularizer,
    gradient_penalty,
    orthogonality_regularizer,
    COMMON_REGULARIZERS,
)

from .modules import GroupSum

from .warnings import (
    DiffLUTWarning,
    PerformanceWarning,
    ConfigurationWarning,
    configure_warnings,
    enable_all_warnings,
    disable_all_warnings,
    disable_cuda_warnings,
    enable_strict_mode,
    suppress_warnings,
)

__all__ = [
    # Regularizers
    'l1_weights',
    'l2_weights',
    'l1_lut_only',
    'l2_lut_only',
    'entropy_regularizer',
    'sparsity_regularizer',
    'gradient_penalty',
    'orthogonality_regularizer',
    'COMMON_REGULARIZERS',
    # Modules
    'GroupSum',
    # Warning utilities
    'DiffLUTWarning',
    'PerformanceWarning',
    'ConfigurationWarning',
    'configure_warnings',
    'enable_all_warnings',
    'disable_all_warnings',
    'disable_cuda_warnings',
    'enable_strict_mode',
    'suppress_warnings',
]
