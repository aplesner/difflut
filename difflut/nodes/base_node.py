import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Callable
import warnings
from ..registry import REGISTRY

class CustomNodeFunction(torch.autograd.Function):
    """
    Custom autograd function that handles custom input gradients while preserving parameter gradients.
    """
    @staticmethod
    def forward(ctx, x, node, *params):
        """
        Forward pass - save what we need for backward
        Args:
            x: Input tensor
            node: The node instance
            *params: All parameters of the node (to track gradients)
        """
        ctx.save_for_backward(x, *params)
        ctx.node = node
        ctx.num_params = len(params)
        
        # Execute forward pass
        if node.training:
            output = node.forward_train(x)
        else:
            output = node.forward_eval(x)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass - computes both input and parameter gradients.
        Uses custom backward for input gradients if provided.
        Parameter gradients are computed via autograd on the forward pass output.
        """
        saved_tensors = ctx.saved_tensors
        x = saved_tensors[0]
        params = saved_tensors[1:] if len(saved_tensors) > 1 else []
        node = ctx.node
        
        grad_x = None
        grad_params = [None] * len(params)
        
        # Step 1: Compute input gradients using custom backward if available
        if ctx.needs_input_grad[0]:
            try:
                grad_x = node.backward(grad_output.contiguous(), x)
            except (NotImplementedError, AttributeError):
                grad_x = None
            
            # Fallback to autograd if custom backward not provided or returns None
            if grad_x is None:
                x_auto = x.detach().requires_grad_(True)
                with torch.enable_grad():
                    if node.training:
                        output = node.forward_train(x_auto)
                    else:
                        output = node.forward_eval(x_auto)
                    
                    if output.requires_grad:
                        grad_x = torch.autograd.grad(
                            outputs=output,
                            inputs=x_auto,
                            grad_outputs=grad_output.contiguous(),
                            retain_graph=False,
                            create_graph=False
                        )[0]
                    else:
                        grad_x = torch.zeros_like(x)
        
        # Step 2: Compute parameter gradients using autograd
        # Recompute forward pass with gradients enabled for parameters
        if len(params) > 0:
            x_for_params = x.detach().requires_grad_(False)
            with torch.enable_grad():
                if node.training:
                    output_for_params = node.forward_train(x_for_params)
                else:
                    output_for_params = node.forward_eval(x_for_params)
                
                # Backward pass for parameters
                if output_for_params.requires_grad:
                    grad_params = torch.autograd.grad(
                        outputs=output_for_params,
                        inputs=params,
                        grad_outputs=grad_output.contiguous(),
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True
                    )
                    # Convert None to zeros for unused parameters
                    grad_params = [g if g is not None else torch.zeros_like(p) 
                                   for g, p in zip(grad_params, params)]
        
        return (grad_x, None) + tuple(grad_params)


class BaseNode(nn.Module, ABC):
    """
    Abstract base class for all LUT nodes with automatic gradient handling.
    
    Default behavior:
    - Initialization: gaussian_init (if no init_fn provided)
    - Regularization: None (if no regularizers provided)
    """
    
    def __init__(self, input_dim: list = None, output_dim: list = None, 
                 use_surrogate: bool = True, 
                 init_fn: Optional[Callable] = None, 
                 init_kwargs: dict = None,
                 regularizers: dict = None):
        """
        Args:
            input_dim: Input dimensions as a list (e.g., [6] for 6 inputs, [6, 6] for 6x6 inputs)
            output_dim: Output dimensions as a list (e.g., [1] for single output, [4] for 4 outputs)
            use_surrogate: Whether to use surrogate gradients (if implemented)
            init_fn: Optional initialization function. Defaults to gaussian_init if not provided.
            init_kwargs: Optional dict of kwargs to pass to the initializer function
            regularizers: Dict of regularization to apply.
                         Format: {"name": [weight, kwargs], ...}
                         where "name" is a registered regularizer name, weight is a float,
                         and kwargs is an optional dict of parameters for the regularizer.
                         Example: {"l1": [0.01, {"num_samples": 100}], "spectral": [0.001]}
        """
        super().__init__()
        
        # Set defaults if not provided
        self.input_dim = input_dim if input_dim is not None else [1]
        self.output_dim = output_dim if output_dim is not None else [1]
        
        # For backward compatibility, compute num_inputs as the product of input dimensions
        self.num_inputs = int(torch.prod(torch.tensor(self.input_dim)).item())
        self.num_outputs = int(torch.prod(torch.tensor(self.output_dim)).item())
        
        # Warn about potentially large LUT sizes
        lut_size = 2 ** self.num_inputs
        if self.num_inputs > 10:
            warnings.warn(
                f"Node initialized with {self.num_inputs} inputs, resulting in a LUT size of {lut_size}. "
                f"Large LUT sizes (>1024 entries) can cause memory issues and slow training. "
                f"Consider using fewer inputs (n<=10) or using grouped/residual layers to reduce complexity.",
                RuntimeWarning,
                stacklevel=3
            )
        
        self.use_surrogate = use_surrogate
        self.init_kwargs = init_kwargs or {}
        
        # Set default initializer to gaussian_init if not provided
        if init_fn is None:
            init_fn = REGISTRY.get_initializer("gaussian")
        self.init_fn = init_fn
        
        # Validate and normalize regularizers format
        self.regularizers = self._validate_regularizers(regularizers)
        
        # Apply initialization
        self.init_fn(self, **self.init_kwargs)
    
    def _validate_regularizers(self, regularizers: dict) -> dict:
        """Validate and normalize regularizers format to internal [fn, weight, kwargs] format"""
        if not regularizers:
            return {}
        
        validated = {}
        for name, value in regularizers.items():
            # Ensure value is list/tuple
            if not isinstance(value, (list, tuple)):
                warnings.warn(
                    f"Regularizer '{name}' should be [weight] or [weight, kwargs], "
                    f"but got {type(value).__name__}. Skipping. "
                    f"Example: regularizers={{'l1': [0.01]}} or {{'l1': [0.01, {{'num_samples': 100}}]}}",
                    UserWarning,
                    stacklevel=3
                )
                continue
            
            # Parse the regularizer value
            if len(value) == 1:
                # Format: [weight]
                weight = value[0]
                kwargs = {}
            elif len(value) == 2:
                # Format: [weight, kwargs]
                weight, kwargs = value
                if not isinstance(kwargs, dict):
                    warnings.warn(
                        f"Regularizer '{name}' kwargs should be a dict, got {type(kwargs).__name__}. Using empty dict.",
                        UserWarning,
                        stacklevel=3
                    )
                    kwargs = {}
            else:
                warnings.warn(
                    f"Regularizer '{name}' has {len(value)} elements, expected 1 or 2 [weight] or [weight, kwargs]. "
                    f"Skipping.",
                    UserWarning,
                    stacklevel=3
                )
                continue
            
            # Lookup regularizer function from registry
            try:
                reg_fn = REGISTRY.get_regularizer(name)
                validated[name] = [reg_fn, weight, kwargs]
            except ValueError as e:
                warnings.warn(
                    f"Regularizer '{name}' not found in registry. Available: {REGISTRY.list_regularizers()}. "
                    f"Skipping.",
                    UserWarning,
                    stacklevel=3
                )
        
        return validated
    
    @abstractmethod
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass during training mode."""
        pass
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass during evaluation mode.
        Default: binarize output of forward_train at threshold 0.5
        Override this method if you need different evaluation behavior.
        """
        return (self.forward_train(x) > 0.5).float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Main forward pass that uses custom autograd function.
        Automatically detects if backward() is implemented, otherwise falls back to autograd.
        """
        # Check if node has a custom backward implementation
        has_custom_backward = False
        try:
            # Check if backward method is overridden
            backward_method = getattr(self.__class__, 'backward', None)
            if backward_method is not None:
                # Check if it's not just the base class version
                base_backward = getattr(BaseNode, 'backward', None)
                has_custom_backward = backward_method is not base_backward
        except:
            pass
        
        # If no custom backward, just use the forward_train/eval directly
        # This allows autograd to work naturally with model parameters
        if not has_custom_backward:
            if self.training:
                return self.forward_train(x)
            else:
                return self.forward_eval(x)
        else:
            # Only use custom autograd function if we have custom backward
            # Pass all parameters explicitly to track gradients
            params = list(self.parameters())
            return CustomNodeFunction.apply(x, self, *params)
    
    def backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Compute custom gradient with respect to input.
        Override this to provide custom gradients (e.g., surrogate gradients).
        
        Args:
            grad_output: Gradient from downstream
            x: Original input tensor
            
        Returns:
            Gradient with respect to input (or None to use autograd)
        """
        return None  # Default: use autograd
    
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
                if len(reg_config) == 3:
                    reg_fn, weight, kwargs = reg_config
                    reg_value = reg_fn(self, **kwargs)
                else:
                    # Fallback for old format [fn, weight]
                    reg_fn, weight = reg_config[:2]
                    reg_value = reg_fn(self)
                reg = reg + weight * reg_value
            except Exception as e:
                print(f"Warning: Regularizer '{name}' failed with error: {e}")
        
        return reg
    
    def _builtin_regularization(self) -> torch.Tensor:
        """
        Built-in regularization for the node (default: none).
        Override this in subclasses to provide node-specific regularization.
        """
        device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
        return torch.tensor(0.0, device=device)
    
    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare input tensor by handling different input dimensions.
        Squeezes middle dimension if input is 3D.
        
        Args:
            x: Input tensor of shape (batch_size, num_inputs) or (batch_size, 1, num_inputs)
        
        Returns:
            Tensor of shape (batch_size, num_inputs)
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        return x
    
    def _prepare_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Prepare output tensor by squeezing if single output dimension.
        
        Args:
            output: Output tensor of shape (batch_size, num_outputs)
        
        Returns:
            Tensor of shape (batch_size,) if num_outputs==1, else (batch_size, num_outputs)
        """
        if self.num_outputs == 1 and output.dim() > 1:
            output = output.squeeze(-1)
        return output
    
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