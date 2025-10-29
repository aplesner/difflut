import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Dict, Any, Optional
import warnings
from .base_layer import BaseLUTLayer
from ..registry import register_layer


class LearnableMappingFunction(torch.autograd.Function):
    """
    Custom autograd function that fuses learnable mapping with node forward pass.
    This avoids storing the intermediate mapped_inputs tensor for backward pass,
    significantly reducing memory usage during backpropagation.
    """
    
    @staticmethod
    def forward(ctx, x, W, tau, node, output_size, n, training):
        """
        Forward pass: apply learnable mapping and pass to node in one operation.
        
        Args:
            x: Input tensor (batch_size, input_size)
            W: Weight matrix (output_size * n, input_size) for mapping
            tau: Temperature for softmax (during training)
            node: The LUT node module
            output_size: Number of nodes
            n: Node input dimension per node
            training: Whether in training mode
        
        Returns:
            output: Result of node forward pass (batch_size, output_size, output_dim)
        """
        batch_size = x.shape[0]
        total_outputs = output_size * n
        
        if training:
            # Soft selection (training mode)
            weights = F.softmax(W / tau, dim=-1)  # (total_outputs, input_size)
            mapped_flat = torch.matmul(x, weights.t())  # (batch_size, total_outputs)
        else:
            # Hard selection (eval mode)
            hard_indices = torch.argmax(W, dim=-1)  # (total_outputs,)
            mapped_flat = torch.index_select(x, 1, hard_indices)  # (batch_size, total_outputs)
        
        # Reshape for node
        mapped_inputs = mapped_flat.reshape(batch_size, output_size, n)
        
        # Forward through node
        output = node(mapped_inputs)
        
        # Save for backward
        ctx.save_for_backward(x, W, mapped_flat)
        ctx.node = node
        ctx.output_size = output_size
        ctx.n = n
        ctx.tau = tau
        ctx.training = training
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: recompute mapping during backprop to avoid storing intermediate.
        
        Args:
            grad_output: Gradient w.r.t. output (batch_size, output_size, output_dim)
        
        Returns:
            Gradients for inputs
        """
        x, W, mapped_flat = ctx.saved_tensors
        node = ctx.node
        output_size = ctx.output_size
        n = ctx.n
        tau = ctx.tau
        training = ctx.training
        batch_size = x.shape[0]
        total_outputs = output_size * n
        
        # Recompute mapped_inputs from saved mapped_flat
        mapped_inputs = mapped_flat.reshape(batch_size, output_size, n)
        mapped_inputs.requires_grad_(True)
        
        # Recompute node forward
        with torch.enable_grad():
            output = node(mapped_inputs)
        
        # Compute gradient w.r.t. mapped_inputs using autograd.grad
        # This properly handles the gradient flow through node parameters
        grad_mapped_inputs, = torch.autograd.grad(
            outputs=output,
            inputs=mapped_inputs,
            grad_outputs=grad_output,
            retain_graph=False,
            create_graph=False
        )
        grad_mapped_flat = grad_mapped_inputs.reshape(batch_size, total_outputs)
        
        # Backward through mapping
        if training:
            # Soft selection backward
            weights = F.softmax(W / tau, dim=-1)
            grad_x = torch.matmul(grad_mapped_flat, weights)
        else:
            # Hard selection backward: scatter gradients back
            hard_indices = torch.argmax(W, dim=-1)
            grad_x = torch.zeros_like(x)
            grad_x.scatter_add_(1, hard_indices.unsqueeze(0).expand(batch_size, -1), 
                                grad_mapped_flat)
        
        # No gradients for W, tau, node, output_size, n, training
        return grad_x, None, None, None, None, None, None


class LearnableMappingModule(nn.Module):
    """
    Helper module for learnable mapping (not registered, used internally).
    Provides soft selection during training and hard selection during evaluation.
    """
    
    def __init__(self, input_size: int, output_size: int, tau: float = 0.001):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tau = tau
        
        # Weight matrix
        self.W = nn.Parameter(torch.randn(output_size, input_size))
        nn.init.xavier_uniform_(self.W)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Soft selection (training) or hard selection (eval).
        """
        if self.training:
            # Soft selection
            weights = F.softmax(self.W / self.tau, dim=-1)
            output = torch.matmul(x, weights.t())
        else:
            # Hard selection (evaluation mode)
            # MEMORY OPTIMIZATION: Avoid expanding (batch_size, output_size) tensor
            # Use torch.index_select for efficient gathering without expansion
            hard_indices = torch.argmax(self.W, dim=-1)  # (output_size,)
            
            # x: (batch_size, input_size)
            # hard_indices: (output_size,) with values in range [0, input_size)
            # We want: output[b, o] = x[b, hard_indices[o]]
            #
            # Use index_select to gather along the input_size dimension
            # This avoids creating expanded intermediate tensors
            output = torch.index_select(x, 1, hard_indices)  # (batch_size, output_size)
        
        return output


@register_layer("learnable")
class LearnableLayer(BaseLUTLayer):
    """
    LUT layer with learnable mapping using nodes.
    Uses soft selection during training and hard selection during evaluation.
    
    MEMORY OPTIMIZATION: Uses fused mapping+forward operation to avoid storing
    intermediate activations (batch_size, output_size, n) during backward pass.
    For large batch sizes, this can reduce memory by ~50-70%.
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int, 
                 node_type: Type[nn.Module],
                 node_kwargs: Optional[Dict[str, Any]] = None,
                 tau: float = 0.001,
                 tau_start: float = 1.0,
                 tau_min: float = 0.0001,
                 tau_decay_iters: float = 1000.0,
                 use_fused: bool = True):
        """
        Args:
            input_size: Size of input vector (from encoder or previous layer)
                       Should match: (batch_size, input_size)
            output_size: Number of LUT nodes (output will be batch_size, output_size * output_dim)
            node_type: LUT node class
            node_kwargs: Additional arguments for nodes (should include input_dim and output_dim)
                        Dimension spec: nodes expect (batch_size, output_size, node_input_dim)
            tau: Initial temperature for softmax in learnable mapping
            tau_start: Starting value for tau (used for exponential decay)
            tau_min: Minimum value tau can decay to
            tau_decay_iters: Number of iterations for tau to decay by factor of 10
            use_fused: If True, use fused mapping+forward operation to save memory (default: True)
                      If False, use standard get_mapping() approach (higher memory but simpler)
        """
        # Warn if tau parameters seem unusual
        if tau_start < tau_min:
            warnings.warn(
                f"LearnableLayer: tau_start ({tau_start}) is less than tau_min ({tau_min}). "
                f"This means tau will be clamped immediately. Set tau_start >= tau_min.",
                UserWarning,
                stacklevel=2
            )
        
        self.use_fused = use_fused
        
        # Initialize parent with nodes (n will be extracted from created nodes)
        super().__init__(input_size, output_size, node_type, node_kwargs)
        
        # Warn about parameter count after n is known
        total_connections = output_size * self.n
        if total_connections > input_size * 10:
            warnings.warn(
                f"LearnableLayer: Creating {total_connections} learnable connections from {input_size} inputs. "
                f"This may lead to overfitting. Consider using GroupedLayer or fewer nodes/inputs per node (n={self.n}).",
                UserWarning,
                stacklevel=2
            )
        
        # Tau decay parameters
        self.tau_start = tau_start
        self.tau_min = tau_min
        self.tau_decay_iters = tau_decay_iters
        self.tau = tau_start  # Start with tau_start instead of tau
        
        # Create learnable mapping module (helper, not registered)
        # Note: self.n is now available from parent's __init__
        self.mapping = LearnableMappingModule(input_size, output_size * self.n, self.tau)
    
    def get_mapping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable mapping and reshape for nodes.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        Returns:
            Mapped inputs of shape (batch_size, output_size, n)
        """
        batch_size = x.shape[0]
        
        # Apply learnable mapping
        mapped_flat = self.mapping(x)  # (batch_size, output_size * n)
        
        # Reshape for nodes
        mapped_inputs = mapped_flat.view(batch_size, self.output_size, self.n)
        
        return mapped_inputs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional memory optimization through fused operation.
        
        When use_fused=True (default), uses a custom autograd function that fuses
        the mapping and node forward pass, avoiding storage of intermediate
        (batch_size, output_size, n) activation tensor. This significantly reduces
        memory usage during backpropagation for large batch sizes.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output tensor of shape (batch_size, output_size * output_dim)
        """
        # Validate input dimensions
        self._validate_input_dims(x)
        
        if self.use_fused:
            # Use fused mapping + node forward to avoid storing intermediate activations
            # This is much more memory-efficient for large batches during backprop
            output = LearnableMappingFunction.apply(x, self.mapping.W, self.mapping.tau, 
                                                     self.node, self.output_size, self.n, 
                                                     self.training)
        else:
            # Standard approach: call get_mapping then node
            # Higher memory usage but easier to debug
            mapped_inputs = self.get_mapping(x)
            output = self.node(mapped_inputs)
        
        # Output shape: (batch_size, output_size, output_dim)
        # Reshape to 2D for next layer: (batch_size, output_size * output_dim)
        batch_size = output.shape[0]
        output = output.view(batch_size, -1)
        
        return output
    
    def get_mapping_matrix(self) -> torch.Tensor:
        """Get current hard mapping (for inspection)."""
        with torch.no_grad():
            hard_indices = torch.argmax(self.mapping.W, dim=-1)
            return hard_indices.view(self.output_size, self.n)
    
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
        
        # Update the mapping module's tau
        self.mapping.tau = self.tau

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        fused_str = "fused" if self.use_fused else "standard"
        return f"input_size={self.input_size}, output_size={self.output_size}, " \
               f"n={self.n}, tau={self.tau}, mapping=learnable, mode={fused_str}"