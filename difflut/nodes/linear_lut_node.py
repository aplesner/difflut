import torch
import torch.nn as nn
import itertools
from typing import Optional, Callable, Dict, Any, Tuple
from .base_node import BaseNode
from ..registry import register_node


@register_node("linear_lut")
class LinearLUTNode(BaseNode):
    """
    Linear LUT Node with differentiable training and hard decisions for eval.
    Uses autograd for gradient computation.
    Now supports per-layer-node weights for better memory access patterns.
    """
    
    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        layer_size: Optional[int] = None,
        init_fn: Optional[Callable[[torch.Tensor], None]] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        regularizers: Optional[Dict[str, Tuple[Callable, float, Dict[str, Any]]]] = None
    ) -> None:
        """
        Args:
            input_dim: Number of inputs (e.g., 6)
            output_dim: Number of outputs (e.g., 1)
            layer_size: Number of parallel nodes in the layer (e.g., 128)
            init_fn: Optional initialization function. Should take (param: torch.Tensor, **kwargs)
            init_kwargs: Keyword arguments for init_fn
            regularizers: Dict of custom regularization functions
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, layer_size=layer_size,
                         regularizers=regularizers, init_fn=init_fn, init_kwargs=init_kwargs)
        
        # Initialize weights with per-layer-node parameters
        # Shape: (layer_size, input_dim, output_dim)
        # Each of the layer_size nodes has its own (input_dim, output_dim) weight matrix
        self.weights = nn.Parameter(torch.randn(self.layer_size, self.num_inputs, self.num_outputs) * 0.1)
        self._apply_init_fn(self.weights, name="weights")

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training: Use sigmoid for differentiability.
        
        Args:
            x: Input tensor of shape (batch_size, layer_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, layer_size, output_dim)
        """
        batch_size, layer_size, input_dim = x.shape
        
        # Verify layer_size matches
        if layer_size != self.layer_size:
            raise ValueError(
                f"Input layer_size {layer_size} does not match node's layer_size {self.layer_size}"
            )
        
        # Use einsum for efficient batched matrix multiplication
        # x: (batch_size, layer_size, input_dim)
        # weights: (layer_size, input_dim, output_dim)
        # output: (batch_size, layer_size, output_dim)
        z = torch.einsum('bli,lio->blo', x, self.weights)
        output = torch.sigmoid(z)
        
        return output

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Discretize by applying Heaviside at 0.5 to forward_train output.
        This makes it behave like a real LUT with binary outputs.
        
        Args:
            x: Input tensor of shape (batch_size, layer_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, layer_size, output_dim)
        """
        batch_size, layer_size, input_dim = x.shape
        
        # Verify layer_size matches
        if layer_size != self.layer_size:
            raise ValueError(
                f"Input layer_size {layer_size} does not match node's layer_size {self.layer_size}"
            )
        
        # Use einsum for efficient batched matrix multiplication
        # x: (batch_size, layer_size, input_dim)
        # weights: (layer_size, input_dim, output_dim)
        # z: (batch_size, layer_size, output_dim)
        z = torch.einsum('bli,lio->blo', x, self.weights)
        
        # Discretize: Heaviside at 0.0
        output = (z >= 0.0).float()
        
        return output

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization by default."""
        return torch.tensor(0.0, device=self.weights.device)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"