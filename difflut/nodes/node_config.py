"""
Configuration class for node parameters.

This module provides a typed, maintainable way to specify node configuration
instead of using raw dictionaries. Supports all common node parameters as well
as node-specific parameters.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, Dict, Any, Union


@dataclass
class NodeConfig:
    """
    Configuration for LUT node initialization.
    
    This class provides type-safe configuration for all node types with proper
    defaults and documentation. It can be converted to/from dictionaries for
    unified API compatibility.
    
    Common Parameters (all nodes):
        input_dim: Number of inputs per node (e.g., 6 for 6-input LUT)
        output_dim: Number of outputs per node (e.g., 1 for single output)
        layer_size: Number of parallel nodes in layer (set by layer, not user)
        regularizers: Dict of regularization functions
        init_fn: Initialization function for parameters
        init_kwargs: Keyword arguments for initialization function
    
    Node-Specific Parameters:
        DWN/DWNStable:
            - use_cuda: Whether to use CUDA kernels
            - alpha: Gradient scaling factor (DWN only)
            - beta: Hamming distance decay (DWN only)
            - clamp_luts: Clamp LUT values to [0,1] (DWN only)
            - gradient_scale: Gradient scaling factor (DWNStable only)
        
        Fourier:
            - use_all_frequencies: Use all 2^n frequency vectors
            - max_amplitude: Maximum amplitude of oscillation
            - use_cuda: Whether to use CUDA kernels
        
        Hybrid:
            - use_cuda: Whether to use CUDA kernels
        
        NeuralLUT:
            - hidden_width: Width of hidden layers
            - depth: Number of MLP layers
            - skip_interval: Interval for skip connections
            - activation: Activation function ('relu' or 'sigmoid')
            - tau_start: Starting temperature value
            - tau_min: Minimum temperature
            - tau_decay_iters: Temperature decay iterations
            - ste: Use Straight-Through Estimator
            - grad_factor: Gradient scaling factor
        
        PolyLUT:
            - degree: Maximum polynomial degree
        
        Probabilistic:
            - temperature: Temperature for sigmoid scaling
            - eval_mode: Evaluation mode ("expectation" or other)
            - use_cuda: Whether to use CUDA kernels
    
    Example:
        ```python
        # Create configuration
        config = NodeConfig(
            input_dim=6,
            output_dim=1,
            init_fn=my_init_fn,
            init_kwargs={'scale': 0.1}
        )
        
        # Use in layer
        layer = RandomLayer(
            input_size=100,
            output_size=50,
            node_type=DWNNode,
            node_kwargs=config
        )
        ```
    """
    
    # ========================================================================
    # Common parameters (used by all nodes via BaseNode)
    # ========================================================================
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    layer_size: Optional[int] = None  # Set automatically by layer
    regularizers: Optional[Dict[str, Any]] = None
    init_fn: Optional[Callable] = None
    init_kwargs: Optional[Dict[str, Any]] = None
    
    # ========================================================================
    # Node-specific parameters (stored in extra_params)
    # ========================================================================
    # These are not explicit fields to keep the dataclass clean, but are
    # stored in extra_params and can be accessed via to_dict()
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.regularizers is None:
            self.regularizers = {}
        if self.init_kwargs is None:
            self.init_kwargs = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for node initialization.
        
        Returns:
            Dictionary with all non-None parameters including extra_params
        """
        # Start with standard dataclass fields
        result = {}
        
        # Add common parameters if not None
        if self.input_dim is not None:
            result['input_dim'] = self.input_dim
        if self.output_dim is not None:
            result['output_dim'] = self.output_dim
        if self.layer_size is not None:
            result['layer_size'] = self.layer_size
        if self.regularizers:
            result['regularizers'] = self.regularizers
        if self.init_fn is not None:
            result['init_fn'] = self.init_fn
        if self.init_kwargs:
            result['init_kwargs'] = self.init_kwargs
        
        # Add extra parameters (node-specific)
        result.update(self.extra_params)
        
        return result
    
    def copy(self) -> 'NodeConfig':
        """Create a deep copy of this configuration."""
        return NodeConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            layer_size=self.layer_size,
            regularizers=self.regularizers.copy() if self.regularizers else None,
            init_fn=self.init_fn,
            init_kwargs=self.init_kwargs.copy() if self.init_kwargs else None,
            extra_params=self.extra_params.copy()
        )
    
    def with_layer_size(self, layer_size: int) -> 'NodeConfig':
        """
        Create a new config with layer_size set.
        
        This is used by layers to inject the layer_size parameter.
        
        Args:
            layer_size: Number of parallel nodes
            
        Returns:
            New NodeConfig with layer_size set
        """
        new_config = self.copy()
        new_config.layer_size = layer_size
        return new_config
    
    def __repr__(self) -> str:
        """String representation showing all parameters."""
        params = []
        if self.input_dim is not None:
            params.append(f"input_dim={self.input_dim}")
        if self.output_dim is not None:
            params.append(f"output_dim={self.output_dim}")
        if self.layer_size is not None:
            params.append(f"layer_size={self.layer_size}")
        if self.init_fn is not None:
            params.append(f"init_fn={self.init_fn.__name__}")
        if self.extra_params:
            extra_str = ", ".join(f"{k}={v}" for k, v in self.extra_params.items())
            params.append(extra_str)
        
        return f"NodeConfig({', '.join(params)})"
