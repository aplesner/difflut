import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from typing import Optional, Callable
from .base_node import BaseNode
from ..registry import register_node


class GradientScalingFunction(torch.autograd.Function):
    """Custom autograd function that scales gradients during backward pass only."""
    
    @staticmethod
    def forward(ctx, input, grad_factor):
        ctx.grad_factor = grad_factor
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.grad_factor, None


@register_node("neurallut")
class NeuralLUTNode(BaseNode):
    """
    NeuraLUT Node that encapsulates a small MLP inside a LUT.
    Uses autograd for gradient computation.
    """
    
    def __init__(self, 
                 input_dim: list = None,
                 output_dim: list = None,
                 hidden_width: int = 16,
                 depth: int = 4,
                 skip_interval: int = 2,
                 init_fn: Optional[Callable] = None,
                 activation: str = 'relu',
                 regularizers: dict = None,
                 tau_start: float = 1.0,
                 tau_min: float = 0.0001,
                 tau_decay_iters: float = 1000.0,
                 ste: bool = False,
                 grad_factor: float = 1.0):
        """
        Args:
            input_dim: Input dimensions as list (e.g., [6])
            output_dim: Output dimensions as list (e.g., [1])
            hidden_width: Width of hidden layers
            depth: Number of layers in the MLP
            skip_interval: Interval for skip connections (0 = no skips)
            init_fn: Optional initialization function
            activation: Activation function ('relu' or 'sigmoid')
            regularizers: Dict of custom regularization functions
            tau: Initial output scaling factor (division) for backward pass (default: 1.0)
            tau_start: Starting value for tau exponential decay (default: 1.0)
            tau_min: Minimum value tau can decay to (default: 0.0001)
            tau_decay_iters: Number of iterations for tau to decay by factor of 10 (default: 1000.0)
            ste: Whether to use Straight-Through Estimator (default: False)
            grad_factor: Gradient scaling factor for backward pass (default: 1.0)
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, regularizers=regularizers)
        self.output_dim = output_dim
        self.hidden_width = hidden_width
        self.depth = depth
        self.skip_interval = skip_interval
        self.activation_type = activation
        self.grad_factor = grad_factor
        self.ste = ste
        # Tau decay parameters
        self.tau_start = tau_start
        self.tau_min = tau_min
        self.tau_decay_iters = tau_decay_iters
        self.tau = tau_start  # Start with tau_start instead of tau
        # Build the MLP with skip connections
        self._build_network(init_fn)

        self.ste_if = self.ste_forward if self.ste else lambda y_soft, u: y_soft
        
        # LUT storage for evaluation mode
        self.register_buffer('lut_table', None)
        self._lut_computed = False

    def _build_network(self, init_fn: Optional[Callable]):
        """Build the MLP network with skip connections."""
        self.layers = nn.ModuleList()
        
        # Input layer
        in_features = self.num_inputs
        out_features = self.hidden_width if self.depth > 1 else self.num_outputs
        self.layers.append(self._create_linear(in_features, out_features, init_fn))
        
        # Hidden layers
        for i in range(1, self.depth - 1):
            layer = self._create_linear(self.hidden_width, self.hidden_width, init_fn)
            self.layers.append(layer)
        
        # Output layer
        if self.depth > 1:
            self.layers.append(self._create_linear(self.hidden_width, self.num_outputs, init_fn))
        
        # Skip connections
        self.skip_layers = nn.ModuleList()
        if self.skip_interval > 0:
            for i in range(self.depth):
                if i > 0 and i % self.skip_interval == 0:
                    target_dim = self.hidden_width if i < self.depth - 1 else self.num_outputs
                    skip = self._create_linear(self.num_inputs, target_dim, init_fn)
                    self.skip_layers.append(skip)
                else:
                    self.skip_layers.append(None)

    def _create_linear(self, in_features: int, out_features: int, 
                      init_fn: Optional[Callable]) -> nn.Linear:
        """Create and initialize a linear layer."""
        linear = nn.Linear(in_features, out_features, bias=True)
        if init_fn:
            init_fn(linear.weight)
            if linear.bias is not None:
                init_fn(linear.bias)
        else:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        return linear

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function."""
        if self.activation_type == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation_type == 'leakyrelu':
            return F.leaky_relu(x)
        else:
            return F.relu(x)

    def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        x_input = x
        h = x
        
        for i, layer in enumerate(self.layers):
            h = layer(h)
            
            # Add skip connection if available
            if self.skip_interval > 0 and i < len(self.skip_layers):
                if self.skip_layers[i] is not None:
                    h = h + self.skip_layers[i](x_input)
            
            # Apply activation (except for last layer)
            if i < len(self.layers) - 1:
                h = self._activation(h)
        
        return h

    def ste_forward(self, y_soft: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        y_hard = (y_soft > 0.5).float()
        output = y_hard.detach() - y_soft.detach() + y_soft
        return output


    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass during training with binary rounding and STE."""
        x = self._prepare_input(x)
        
        # MLP forward + sigmoid
        logits = self._mlp_forward(x)
        
        u = torch.rand_like(logits)
        y_soft = torch.sigmoid((logits + torch.log(u) - torch.log(1 - u)) / self.tau)
        output = self.ste_if(y_soft, u)
        
        # Apply gradient scaling using custom autograd function
        # This only affects gradients, not forward pass values
        output = GradientScalingFunction.apply(output, torch.tensor(self.grad_factor, device=output.device))
        
        return self._prepare_output(output)

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation: Discretize by applying Heaviside at 0.5 to forward_train output.
        This makes it behave like a real LUT with binary outputs.
        """
        x = self._prepare_input(x)
        
        # Compute same as forward_train (MLP + sigmoid)
        output = self._mlp_forward(x)
        output = (output >= 0.0).float()
        
        return self._prepare_output(output)

    def _precompute_lut(self):
        """Precompute the LUT for evaluation."""
        device = next(self.parameters()).device
        num_entries = 2 ** self.num_inputs
        
        lut = torch.zeros((num_entries, self.num_outputs), device=device)
        
        with torch.no_grad():
            for i, bits in enumerate(itertools.product([0, 1], repeat=self.num_inputs)):
                x = torch.tensor(bits, dtype=torch.float32, device=device).unsqueeze(0)
                output = self._mlp_forward(x)
                # Apply sigmoid to match training forward pass
                output = torch.sigmoid(output)
                lut[i] = output.squeeze(0)
        
        self.lut_table = lut
        self._lut_computed = True

    def _builtin_regularization(self) -> torch.Tensor:
        """No built-in regularization by default."""
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def update_tau(self, iteration: int):
        """
        Update tau using exponential decay.
        
        Args:
            iteration: Current training iteration
        """
        # Calculate decay factor: tau = tau_start * 10^(-iteration / tau_decay_iters)
        # This means tau decays by a factor of 10 every tau_decay_iters iterations
        decay_factor = 10.0 ** (-iteration / self.tau_decay_iters)
        self.tau = max(self.tau_start * decay_factor, self.tau_min)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, " \
               f"hidden_width={self.hidden_width}, depth={self.depth}, " \
               f"skip_interval={self.skip_interval}, activation={self.activation_type}, " \
               f"grad_factor={self.grad_factor}", \
               f"tau={self.tau:.4f}, tau_start={self.tau_start}, " \
               f"tau_min={self.tau_min}, tau_decay_iters={self.tau_decay_iters}"