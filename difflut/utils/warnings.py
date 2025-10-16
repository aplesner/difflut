"""
Warning utilities for DiffLUT package.

This module provides utilities for managing warnings throughout the DiffLUT package.
Users can control warning behavior using standard Python warnings module.

Examples:
    # Suppress all DiffLUT warnings
    >>> import warnings
    >>> warnings.filterwarnings('ignore', module='difflut')
    
    # Suppress only RuntimeWarnings (e.g., CUDA unavailable)
    >>> warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut')
    
    # Show warnings only once (default)
    >>> warnings.filterwarnings('once', module='difflut')
    
    # Treat warnings as errors for strict checking
    >>> warnings.filterwarnings('error', module='difflut')
"""

import warnings
from typing import Optional


class DiffLUTWarning(UserWarning):
    """Base warning class for DiffLUT-specific warnings."""
    pass


class PerformanceWarning(RuntimeWarning):
    """Warning for performance-related issues."""
    pass


class ConfigurationWarning(UserWarning):
    """Warning for suboptimal or incorrect configuration."""
    pass


def warn_once(message: str, category: type = UserWarning, stacklevel: int = 2):
    """
    Issue a warning that will only be shown once per location.
    
    Args:
        message: Warning message
        category: Warning category class
        stacklevel: Stack level for warning source
    """
    warnings.warn(message, category=category, stacklevel=stacklevel + 1)


def warn_cuda_unavailable(module_name: str, extension_name: str, stacklevel: int = 2):
    """
    Standard warning for unavailable CUDA extensions.
    
    Args:
        module_name: Name of the module (e.g., "DWNNode")
        extension_name: Name of the CUDA extension (e.g., "efd_cuda")
        stacklevel: Stack level for warning source
    """
    warnings.warn(
        f"{module_name}: CUDA extension '{extension_name}' not available. "
        f"Using slower CPU fallback. For better performance, compile the extension: "
        f"'cd difflut && python setup.py install'. "
        f"To suppress: warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut')",
        PerformanceWarning,
        stacklevel=stacklevel + 1
    )


def warn_large_lut(num_inputs: int, lut_size: int, stacklevel: int = 2):
    """
    Warning for potentially problematic large LUT sizes.
    
    Args:
        num_inputs: Number of inputs to the LUT
        lut_size: Total LUT size (2^num_inputs)
        stacklevel: Stack level for warning source
    """
    warnings.warn(
        f"LUT with {num_inputs} inputs creates {lut_size} entries. "
        f"Large LUTs (>1024) may cause memory issues and slow training. "
        f"Consider using fewer inputs (n<=10) or grouped/residual layers.",
        ConfigurationWarning,
        stacklevel=stacklevel + 1
    )


def warn_parameter_count(param_count: int, threshold: int = 100000, stacklevel: int = 2):
    """
    Warning for excessive parameter counts.
    
    Args:
        param_count: Number of parameters
        threshold: Threshold for warning
        stacklevel: Stack level for warning source
    """
    if param_count > threshold:
        warnings.warn(
            f"Layer has {param_count:,} parameters (>{threshold:,}). "
            f"This may lead to overfitting or memory issues. "
            f"Consider using GroupedLayer or reducing layer size.",
            ConfigurationWarning,
            stacklevel=stacklevel + 1
        )


def warn_encoder_not_fitted(encoder_class: str, stacklevel: int = 2):
    """
    Warning/error for using unfitted encoder.
    
    Args:
        encoder_class: Name of the encoder class
        stacklevel: Stack level for warning source
    """
    raise RuntimeError(
        f"{encoder_class} must be fitted before encoding. "
        f"Call fit() first with your training data. "
        f"Example: encoder.fit(train_data).encode(test_data)"
    )


def configure_warnings(
    action: str = 'default',
    category: Optional[type] = None,
    module_pattern: str = 'difflut'
):
    """
    Configure warning filters for DiffLUT.
    
    Args:
        action: Warning action - 'default', 'ignore', 'always', 'error', 'once'
        category: Specific warning category to filter (None for all)
        module_pattern: Module pattern to match (default: 'difflut' for all)
    
    Examples:
        >>> # Ignore all DiffLUT warnings
        >>> configure_warnings('ignore')
        
        >>> # Show only RuntimeWarnings
        >>> configure_warnings('default', category=RuntimeWarning)
        
        >>> # Ignore CUDA warnings specifically
        >>> configure_warnings('ignore', module_pattern='difflut.nodes')
    """
    if category is None:
        warnings.filterwarnings(action, module=module_pattern)
    else:
        warnings.filterwarnings(action, category=category, module=module_pattern)


def suppress_warnings(func):
    """
    Decorator to suppress warnings for a specific function.
    
    Example:
        >>> @suppress_warnings
        >>> def my_function():
        >>>     # Warnings from DiffLUT will be suppressed here
        >>>     pass
    """
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='difflut')
            return func(*args, **kwargs)
    return wrapper


# Preset warning configurations
def enable_all_warnings():
    """Enable all DiffLUT warnings (useful for debugging)."""
    configure_warnings('always')


def disable_all_warnings():
    """Disable all DiffLUT warnings."""
    configure_warnings('ignore')


def disable_cuda_warnings():
    """Disable only CUDA-related performance warnings."""
    configure_warnings('ignore', category=PerformanceWarning)


def enable_strict_mode():
    """Treat all DiffLUT warnings as errors."""
    configure_warnings('error')


# Default configuration: warnings shown once per location
# Users can reconfigure using the functions above or Python's warnings module
