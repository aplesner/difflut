"""
Warning utilities for DiffLUT package.

This module provides utilities for managing warnings throughout the DiffLUT package.
Users can control warning behavior using standard Python warnings module.

Examples:
    # Suppress all DiffLUT warnings
    >>> import warnings
    >>> warnings.filterwarnings('ignore', module='difflut')

    # Suppress only DefaultValueWarnings
    >>> warnings.filterwarnings('ignore', category=DefaultValueWarning, module='difflut')

    # Show warnings only once (default)
    >>> warnings.filterwarnings('once', module='difflut')

    # Treat warnings as errors for strict checking
    >>> warnings.filterwarnings('error', module='difflut')
"""

import warnings
from typing import Any, Optional, Type


class DiffLUTWarning(UserWarning):
    """Base warning class for DiffLUT-specific warnings."""

    pass


class PerformanceWarning(RuntimeWarning):
    """Warning for performance-related issues."""

    pass


class ConfigurationWarning(UserWarning):
    """Warning for suboptimal or incorrect configuration."""

    pass


class DefaultValueWarning(ConfigurationWarning):
    """Warning for use of default configuration values."""

    pass


class CUDAWarning(PerformanceWarning):
    """Warning for CUDA availability and optimization issues."""

    pass


def warn_default_value(param_name: str, param_value: Any, stacklevel: int = 2) -> None:
    """
    Issue a warning when a default value is used for an optional parameter.

    This is the primary warning utility used throughout DiffLUT to alert users
    when parameters are not explicitly provided and default values are being used.

    Args:
        param_name: Name of the parameter
        param_value: Default value being used
        stacklevel: Stack level for warning source (default 2 for direct calls,
                   3 for nested function calls)

    Example:
        >>> if param is None:
        >>>     param = DEFAULT_VALUE
        >>>     warn_default_value("param", param, stacklevel=2)
    """
    warnings.warn(
        f"Parameter '{param_name}' was not provided, using default value: {param_value}",
        category=DefaultValueWarning,
        stacklevel=stacklevel + 1,
    )


def configure_warnings(
    action: str = "default",
    category: Optional[Type[Warning]] = None,
    module_pattern: str = "difflut",
) -> None:
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


def suppress_warnings(func: Any) -> Any:
    """
    Decorator to suppress warnings for a specific function.

    Example:
        >>> @suppress_warnings
        >>> def my_function():
        >>>     # Warnings from DiffLUT will be suppressed here
        >>>     pass
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="difflut")
            return func(*args, **kwargs)

    return wrapper


# Preset warning configurations
def enable_all_warnings() -> None:
    """Enable all DiffLUT warnings (useful for debugging)."""
    configure_warnings("always")


def disable_all_warnings() -> None:
    """Disable all DiffLUT warnings."""
    configure_warnings("ignore")


def disable_cuda_warnings() -> None:
    """Disable only CUDA-related performance warnings."""
    configure_warnings("ignore", category=PerformanceWarning)


def enable_strict_mode() -> None:
    """Treat all DiffLUT warnings as errors."""
    configure_warnings("error")


# Default configuration: warnings shown once per location
# Users can reconfigure using the functions above or Python's warnings module
