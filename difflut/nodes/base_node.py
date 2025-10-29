import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Callable
import warnings


class BaseNode(nn.Module, ABC):
    """
    Abstract base class for all LUT nodes with automatic gradient handling
    """
    
    def __init__(self, input_dim: int = None, output_dim: int = None, 
                 layer_size: int = None,
                 regularizers: dict = None,
                 init_fn: Optional[Callable] = None, init_kwargs: dict = None):
        """
        Args:
            input_dim: Number of inputs (e.g., 6 for 6 inputs)
            output_dim: Number of outputs (e.g., 1 for single output, 4 for 4 outputs)
            layer_size: Number of parallel nodes in the layer (e.g., 128 for 128 parallel nodes)
                       This controls weight sharing - each node in layer_size will have separate parameters.
            regularizers: Dict of regularization functions to apply.
                         Format: {"name": [reg_fn, weight, kwargs], ...}
                         where reg_fn is callable(node) -> scalar tensor,
                               weight is float, and kwargs is dict.
            init_fn: Optional initialization function for parameters.
                    Should accept (parameter: torch.Tensor, **kwargs) and modify in-place.
                    This is passed to subclasses for their use - BaseNode does NOT apply it.
            init_kwargs: Optional dict of kwargs to pass to the initializer function
        """
        super().__init__()
        
        # Set defaults if not provided
        self.input_dim = input_dim if input_dim is not None else 6
        self.output_dim = output_dim if output_dim is not None else 1
        self.layer_size = layer_size if layer_size is not None else 1
        
        # Validate input_dim
        if not isinstance(self.input_dim, int) or self.input_dim <= 0:
            raise ValueError(
                f"input_dim must be a positive integer, but got {self.input_dim}. "
                f"Example: input_dim=6"
            )
        
        # Validate layer_size
        if not isinstance(self.layer_size, int) or self.layer_size <= 0:
            raise ValueError(
                f"layer_size must be a positive integer, but got {self.layer_size}. "
                f"Example: layer_size=128"
            )
        
        if self.input_dim > 10:
            warnings.warn(
                f"input_dim={self.input_dim} is quite large. "
                f"LUT nodes with >10 inputs may have exponentially large memory requirements (2^{self.input_dim} entries). "
                f"Consider using smaller input dimensions or splitting inputs across multiple layers.",
                UserWarning,
                stacklevel=2
            )
        
        # Validate output_dim
        if not isinstance(self.output_dim, int) or self.output_dim <= 0:
            raise ValueError(
                f"output_dim must be a positive integer, but got {self.output_dim}. "
                f"Example: output_dim=1"
            )
        
        if self.output_dim > 10:
            warnings.warn(
                f"output_dim={self.output_dim} is quite large. "
                f"This may increase memory requirements significantly.",
                UserWarning,
                stacklevel=2
            )
        
        # Validate and store regularizers
        self.regularizers = regularizers or {}
        
        # Validate init_fn
        if init_fn is not None:
            if not callable(init_fn):
                raise TypeError(
                    f"init_fn must be callable, but got {type(init_fn).__name__}. "
                    f"Signature should be: init_fn(parameter: torch.Tensor, **kwargs) -> None"
                )
        
        self.init_fn = init_fn
        self.init_kwargs = init_kwargs or {}
        
        # Validate init_kwargs
        if not isinstance(self.init_kwargs, dict):
            raise TypeError(
                f"init_kwargs must be a dict, but got {type(self.init_kwargs).__name__}"
            )
        
        # Validate and normalize regularizers format
        self._validate_regularizers()
    
    @property
    def num_inputs(self) -> int:
        """Get number of inputs."""
        return self.input_dim
    
    @property
    def num_outputs(self) -> int:
        """Get number of outputs."""
        return self.output_dim
    
    def _apply_init_fn(self, param: torch.Tensor, name: str = "parameter") -> None:
        """
        Apply the node's init_fn to a specific parameter.
        Helper method for subclasses to initialize parameters selectively.
        
        Args:
            param: The parameter tensor to initialize
            name: Name of the parameter (for error messages)
            
        Raises:
            RuntimeError: If init_fn is set but fails during execution
        """
        if self.init_fn is None:
            return
        
        try:
            self.init_fn(param, **self.init_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Initialization of '{name}' failed with error: {e}. "
                f"Check that init_fn is compatible with the parameter and init_kwargs are correct."
            )
    
    def _validate_regularizers(self):
        """Validate regularizers format - only accepts [fn, weight, kwargs] format"""
        if not self.regularizers:
            return
        
        validated = {}
        for name, value in self.regularizers.items():
            # Check if it's a list/tuple
            if not isinstance(value, (list, tuple)):
                raise ValueError(
                    f"Regularizer '{name}' must be a list/tuple of [function, weight, kwargs], "
                    f"but got {type(value).__name__}. "
                    f"Example: regularizers={{'l2': [l2_reg, 0.01, {{}}]}} or "
                    f"{{'l2': [l2_reg, 0.01, {{'num_samples': 100}}]}}"
                )
            
            # Check length
            if len(value) != 3:
                raise ValueError(
                    f"Regularizer '{name}' must have exactly 3 elements [function, weight, kwargs], "
                    f"but got {len(value)} elements. "
                    f"Example: regularizers={{'l2': [l2_reg, 0.01, {{}}]}}"
                )
            
            reg_fn, weight, kwargs = value
            
            # Validate function is callable
            if not callable(reg_fn):
                raise TypeError(
                    f"Regularizer '{name}' function must be callable, but got {type(reg_fn).__name__}. "
                    f"The function should take the node as input and return a scalar tensor."
                )
            
            # Validate weight is numeric
            if not isinstance(weight, (int, float)):
                raise TypeError(
                    f"Regularizer '{name}' weight must be numeric (int or float), but got {type(weight).__name__}."
                )
            
            # Validate kwargs is dict
            if not isinstance(kwargs, dict):
                raise TypeError(
                    f"Regularizer '{name}' kwargs must be a dict, but got {type(kwargs).__name__}."
                )
            
            validated[name] = [reg_fn, weight, kwargs]
        
        self.regularizers = validated
    
    def _select_independent_luts(self, output_flat: torch.Tensor, batch_size: int, layer_size: int) -> torch.Tensor:
        """
        Helper method to select independent LUT outputs when output_dim == layer_size.
        
        When a node has output_dim==layer_size, it means we have independent LUTs
        (one for each position). This method selects the appropriate LUT output for each position.
        
        Args:
            output_flat: Tensor of shape (batch*layer_size, output_dim) or (batch, layer_size, output_dim)
            batch_size: Batch size
            layer_size: Number of positions (nodes) in the layer
        
        Returns:
            Tensor of shape (batch, layer_size, 1) with independent LUT outputs selected
        """
        has_independent_luts = (self.output_dim == layer_size)
        
        if not has_independent_luts or self.output_dim == 1:
            # Standard case: all positions share same LUT or output_dim=1
            if output_flat.dim() == 2:
                return output_flat.view(batch_size, layer_size, self.output_dim)
            else:
                return output_flat
        
        # Independent LUTs case: select appropriate column for each position
        if output_flat.dim() == 2:
            # Shape: (batch*layer_size, output_dim)
            output_3d = output_flat.view(batch_size, layer_size, self.output_dim)
        else:
            # Already 3D: (batch, layer_size, output_dim)
            output_3d = output_flat
        
        # Select diagonal elements: output_3d[:, j, j] for each position j
        indices = torch.arange(layer_size, device=output_flat.device)
        output = output_3d[
            torch.arange(batch_size, device=output_flat.device).unsqueeze(1),
            torch.arange(layer_size, device=output_flat.device).unsqueeze(0),
            indices.unsqueeze(0).expand(batch_size, -1)
        ]
        
        # Add last dimension: (batch, layer_size) → (batch, layer_size, 1)
        return output.unsqueeze(-1)
    
    @abstractmethod
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training mode.
        
        Args:
            x: Tensor of shape (batch_size, layer_size, input_dim)
               where layer_size is the number of independent node copies
        
        Returns:
            Tensor of shape (batch_size, layer_size, output_dim)
        """
        pass
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during evaluation mode.
        Default: binarize output of forward_train at threshold 0.5
        Override this method if you need different evaluation behavior.
        
        Args:
            x: Tensor of shape (batch_size, layer_size, input_dim)
        
        Returns:
            Tensor of shape (batch_size, layer_size, output_dim)
        """
        return (self.forward_train(x) > 0.5).float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Main forward pass that automatically dispatches to forward_train or forward_eval.

        Expects 3D batched input of multiple independent node copies:
        - Input shape: (batch_size, layer_size, input_dim)
          where layer_size is the number of independent node copies
        - Output shape: (batch_size, layer_size, output_dim)

        This allows efficient batch processing across all node copies simultaneously.
        """
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_eval(x)

    def forward_with_mapping(self, x: torch.Tensor, mapping_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fused mapping (memory optimization).

        Instead of receiving pre-materialized mapped_inputs (batch, layer_size, input_dim),
        this method receives the raw input and mapping indices, allowing nodes to perform
        indexing inside optimized CUDA kernels and avoid materializing large intermediate tensors.

        Default implementation: materializes mapped_inputs and calls forward().
        Nodes with fused CUDA kernels should override this method for memory efficiency.

        Args:
            x: Input tensor of shape (batch_size, input_size)
               Raw 2D input from encoder or previous layer
            mapping_indices: Mapping indices of shape (layer_size, input_dim)
                           Contains indices into input_size dimension, values in [0, input_size)
                           Defines which inputs to select for each node

        Returns:
            Output tensor of shape (batch_size, layer_size, output_dim)

        Memory Impact:
            - Materialized: Creates (batch, layer_size, input_dim) tensor → high memory
            - Fused (in subclass): Performs indexing in CUDA kernel → minimal memory

        Example:
            # Materialized path (default):
            mapped = x[batch_idx, mapping_indices]  # (batch, layer_size, input_dim)
            output = forward(mapped)

            # Fused path (in DWNNode override):
            output = cuda_kernel(x, mapping_indices, luts)  # No intermediate tensor
        """
        batch_size = x.shape[0]

        # Materialize mapped_inputs using advanced indexing
        # Shape: (batch_size, 1, 1) broadcasts with (layer_size, input_dim)
        batch_idx = torch.arange(batch_size, device=x.device).view(-1, 1, 1)

        # Advanced indexing: x[batch, mapping[layer, dim]]
        # Result: (batch_size, layer_size, input_dim)
        mapped_inputs = x[batch_idx, mapping_indices]

        # Call standard forward pass
        return self.forward(mapped_inputs)
    
    def regularization(self) -> torch.Tensor:
        """
        Compute regularization term for the node.
        Combines built-in regularization with custom regularizers.
        Override _builtin_regularization() in subclasses for node-specific regularization.
        """
        device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
        
        # Start with built-in regularization (default: 0)
        reg = self._builtin_regularization()
        
        # Add custom regularizers
        for name, reg_config in self.regularizers.items():
            try:
                reg_fn, weight, kwargs = reg_config
                reg_value = reg_fn(self, **kwargs)
                reg = reg + weight * reg_value
            except Exception as e:
                raise RuntimeError(
                    f"Regularizer '{name}' failed with error: {e}. "
                    f"Check that the regularizer function is correct and compatible with the node."
                )
        
        return reg
    
    def _builtin_regularization(self) -> torch.Tensor:
        """
        Built-in regularization for the node (default: none).
        Override this in subclasses to provide node-specific regularization.
        
        Returns:
            Scalar tensor with regularization value (default: 0.0)
        """
        device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
        return torch.tensor(0.0, device=device)
    
    def export_bitstream(self) -> list:
        """
        Export node configuration as bitstream for FPGA deployment.
        Default: evaluate all binary input combinations using forward_eval.
        Override this method if you need custom export behavior.
        """
        import itertools
        bitstream = []
        
        with torch.no_grad():
            device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
            
            for bits in itertools.product([0, 1], repeat=self.num_inputs):
                x = torch.tensor([bits], dtype=torch.float32, device=device)
                output = self.forward_eval(x)
                
                # Convert output to binary list
                if output.dim() == 0:
                    bitstream.append(int(output.item() > 0.5))
                else:
                    output_flat = output.flatten()
                    bitstream.extend([int(val.item() > 0.5) for val in output_flat])
        
        return bitstream