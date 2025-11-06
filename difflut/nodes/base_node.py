import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..utils.warnings import DefaultValueWarning, warn_default_value

# Default number of inputs per node if not specified
DEFAULT_NODE_INPUT_DIM: int = 6
# Default number of outputs per node if not specified
DEFAULT_NODE_OUTPUT_DIM: int = 1
# Threshold for warning about large input dimensions
# If input_dim > NODE_INPUT_DIM_WARNING_THRESHOLD, warn about memory
NODE_INPUT_DIM_WARNING_THRESHOLD: int = 10
# Threshold for warning about large output dimensions
# If output_dim > NODE_OUTPUT_DIM_WARNING_THRESHOLD, warn about memory
NODE_OUTPUT_DIM_WARNING_THRESHOLD: int = 10


class BaseNode(nn.Module, ABC):
    """
    Abstract base class for all LUT nodes.

    Architecture:
    - Nodes process 2D tensors: (batch_size, input_dim) → (batch_size, output_dim)
    - Each node instance is independent
    - Layers use nn.ModuleList to manage multiple node instances
    - No 3D tensor processing or layer_size dimension
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        regularizers: Optional[Dict[str, Tuple[Callable, float, Dict[str, Any]]]] = None,
        init_fn: Optional[Callable[[torch.Tensor], None]] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,  # Accept extra kwargs for compatibility but ignore them
    ) -> None:
        """
        Args:
            input_dim: Number of inputs (e.g., 6 for 6 inputs)
            output_dim: Number of outputs (e.g., 1 for single output, 4 for 4 outputs)
            regularizers: Dict of regularization functions to apply.
                         Format: {"name": [reg_fn, weight, kwargs], ...}
                         where reg_fn is callable(node) -> scalar tensor,
                               weight is float, and kwargs is dict.
            init_fn: Optional initialization function for parameters.
                    Should accept (parameter: torch.Tensor, **kwargs) and modify in-place.
                    This is passed to subclasses for their use - BaseNode does NOT apply it.
            init_kwargs: Optional dict of kwargs to pass to the initializer function
            **kwargs: Additional arguments (ignored, for compatibility)
        """
        super().__init__()

        # Set defaults if not provided
        if input_dim is None:
            self.input_dim = DEFAULT_NODE_INPUT_DIM
            warn_default_value("input_dim", self.input_dim, stacklevel=2)
        else:
            self.input_dim = input_dim

        if output_dim is None:
            self.output_dim = DEFAULT_NODE_OUTPUT_DIM
            warn_default_value("output_dim", self.output_dim, stacklevel=2)
        else:
            self.output_dim = output_dim

        # Validate input_dim
        if not isinstance(self.input_dim, int) or self.input_dim <= 0:
            raise ValueError(
                f"input_dim must be a positive integer, but got {self.input_dim}. "
                f"Example: input_dim=6"
            )

        if self.input_dim > NODE_INPUT_DIM_WARNING_THRESHOLD:
            warnings.warn(
                f"input_dim={self.input_dim} is quite large. "
                f"LUT nodes with >{NODE_INPUT_DIM_WARNING_THRESHOLD} inputs may have exponentially large memory requirements (2^{self.input_dim} entries). "
                f"Consider using smaller input dimensions or splitting inputs across multiple layers.",
                UserWarning,
                stacklevel=2,
            )

        # Validate output_dim
        if not isinstance(self.output_dim, int) or self.output_dim <= 0:
            raise ValueError(
                f"output_dim must be a positive integer, but got {self.output_dim}. "
                f"Example: output_dim=1"
            )

        if self.output_dim > NODE_OUTPUT_DIM_WARNING_THRESHOLD:
            warnings.warn(
                f"output_dim={self.output_dim} is quite large. "
                f"This may increase memory requirements significantly.",
                UserWarning,
                stacklevel=2,
            )

        # Validate and store regularizers
        # Note: Only warn if regularizers is truly missing (None), not if explicitly provided as {}
        if regularizers is None:
            self.regularizers = {}
        else:
            self.regularizers = regularizers

        # Validate init_fn
        if init_fn is not None:
            if not callable(init_fn):
                raise TypeError(
                    f"init_fn must be callable, but got {type(init_fn).__name__}. "
                    f"Signature should be: init_fn(parameter: torch.Tensor, **kwargs) -> None"
                )

        self.init_fn = init_fn

        # Handle init_kwargs with default
        # Note: Only warn if init_kwargs is truly missing (None), not if explicitly provided as {}
        if init_kwargs is None:
            self.init_kwargs = {}
        else:
            self.init_kwargs = init_kwargs

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

    def _validate_regularizers(self) -> None:
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

    @abstractmethod
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during training mode.

        Args:
            x: Tensor of shape (batch_size, input_dim) - 2D tensor

        Returns:
            Tensor of shape (batch_size, output_dim) - 2D tensor
        """
        pass

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during evaluation mode.
        Default: binarize output of forward_train at threshold 0.5
        Override this method if you need different evaluation behavior.

        Args:
            x: Tensor of shape (batch_size, input_dim) - 2D tensor

        Returns:
            Tensor of shape (batch_size, output_dim) - 2D tensor
        """
        return (self.forward_train(x) > 0.5).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Main forward pass that automatically dispatches to forward_train or forward_eval.

        Processes 2D tensors:
        - Input shape: (batch_size, input_dim)
        - Output shape: (batch_size, output_dim)

        Each node instance is independent. Layers use nn.ModuleList to manage
        multiple node instances and iterate through them during forward pass.
        """
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_eval(x)

    def regularization(self) -> torch.Tensor:
        """
        Compute regularization term for the node.
        Combines built-in regularization with custom regularizers.
        Override _builtin_regularization() in subclasses for node-specific regularization.

        Returns:
            Scalar tensor with total regularization value
        """
        device = next(self.parameters()).device if list(self.parameters()) else "cpu"

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
        device: torch.device = (
            next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        )
        return torch.tensor(0.0, device=device)

    def export_bitstream(self) -> List[int]:
        """
        Export node configuration as bitstream for FPGA deployment.
        Default: evaluate all binary input combinations using forward_eval.
        Override this method if you need custom export behavior.
        """
        import itertools

        bitstream = []

        with torch.no_grad():
            device = next(self.parameters()).device if list(self.parameters()) else "cpu"

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
