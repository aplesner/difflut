"""
MNIST model zoo implementations.

Models for MNIST classification highlighting different DiffLUT capabilities:
- MNISTLinearSmall: 2-layer FC with 8k nodes, linear LUT nodes
- MNISTDWNSmall: 2-layer FC with 8k nodes, DWN stable nodes (for FPGA export)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path

from difflut import REGISTRY
from difflut.utils import GroupSum
from difflut.nodes.node_config import NodeConfig
from difflut.layers.layer_config import LayerConfig

from .base_model import BaseModel


class MNISTSmallBase(BaseModel):
    """
    Base class for small MNIST models suitable for FPGA export.
    
    Architecture:
    - Input: 28x28 MNIST images (784 pixels)
    - Encoder: Thermometer encoding with 4 bits → 3136 features (784*4)
    - Layer 1: 8000 nodes → 128 output features
    - Layer 2: 128 nodes → 10 output features
    - Classifier: GroupSum for 10 classes
    
    This design prioritizes:
    - Moderate size for fast FPGA compilation
    - Sufficient capacity for MNIST (>95% accuracy)
    - Clear demonstration of layer-wise processing
    """
    
    def __init__(
        self,
        name: str,
        node_type: str = 'linear_lut',
        node_input_dim: int = 4,
        hidden_size: int = 128,
        encoder_bits: int = 4,
        flip_probability: float = 0.0,
        grad_stabilization: str = 'none',
        use_cuda: bool = True,
    ):
        """
        Initialize MNIST small model.
        
        Args:
            name: Model identifier
            node_type: Type of DiffLUT node ('linear_lut', 'dwn_stable', etc.)
            node_input_dim: Number of inputs per node (4 or 6)
            hidden_size: Size of first hidden layer (default 128)
            encoder_bits: Number of thermometer encoder bits per pixel
            flip_probability: Bit flip probability for robustness [0, 1]
            grad_stabilization: Gradient stabilization mode ('none', 'layerwise')
            use_cuda: Whether to use CUDA extensions for DWN nodes
        """
        super().__init__(name=name, input_size=784, num_classes=10)
        
        # Model configuration
        self.node_type = node_type
        self.node_input_dim = node_input_dim
        self.hidden_size = hidden_size
        self.encoder_bits = encoder_bits
        self.use_cuda = use_cuda
        
        # Create encoder (will be fitted on training data)
        from difflut.encoder.thermometer import ThermometerEncoder
        self.encoder = ThermometerEncoder(num_bits=encoder_bits)
        
        # Layer configuration (shared across all layers)
        self.layer_config = LayerConfig(
            flip_probability=flip_probability,
            grad_stabilization=grad_stabilization,
            grad_target_std=1.0,
            grad_subtract_mean=False,
            grad_epsilon=1e-8
        )
        
        # Layers will be built after encoder is fitted
        self.layers = nn.ModuleList()
        self.layer_sizes = []
        
        # Output layer
        self.output_layer = GroupSum(k=self.num_classes, tau=1, use_randperm=False)
    
    def fit_encoder(self, data: torch.Tensor) -> None:
        """
        Fit thermometer encoder on training data.
        
        Args:
            data: Training data tensor (N, ...) - typically MNIST images (N, 28, 28)
        
        Raises:
            RuntimeError: If encoder already fitted
        """
        if self.encoder_fitted:
            raise RuntimeError("Encoder already fitted, cannot fit again")
        
        # Flatten data
        batch_size = data.shape[0]
        if data.dim() > 2:
            data_flat = data.view(batch_size, -1)
        else:
            data_flat = data
        
        # Fit encoder
        print(f"Fitting encoder on {len(data_flat)} samples with shape {data_flat.shape}...")
        self.encoder.fit(data_flat)
        
        # Get encoded size
        sample_encoded = self.encoder.encode(data_flat[:1])
        self.encoded_input_size = sample_encoded.shape[1]
        print(f"Encoded input size: {self.encoded_input_size}")
        
        # Build layers
        self._build_layers()
        self.encoder_fitted = True
    
    def _build_layers(self) -> None:
        """
        Build network layers after encoder is fitted.
        
        Creates:
        - Layer 1: encoded_size → hidden_size (e.g., 3136 → 128)
        - Layer 2: hidden_size → num_classes (128 → 10)
        
        Both layers use the same node type and configuration.
        """
        # Get node and layer classes from registry
        try:
            node_class = REGISTRY.get_node(self.node_type)
            layer_class = REGISTRY.get_layer('random')  # Use random connectivity
        except ValueError as e:
            raise ValueError(f"Failed to get node/layer from registry: {e}")
        
        # Build node kwargs with proper initialization
        node_kwargs = self._build_node_kwargs(
            input_dim=self.node_input_dim,
            current_layer_width=None,
            next_layer_width=self.hidden_size
        )
        
        # Layer 1: Encoded input → Hidden layer
        layer1 = layer_class(
            input_size=self.encoded_input_size,
            output_size=self.hidden_size,
            node_type=node_class,
            node_kwargs=node_kwargs,
            seed=42,
            layer_config=self.layer_config
        )
        self.layers.append(layer1)
        self.layer_sizes.append({
            'type': 'fc',
            'input': self.encoded_input_size,
            'output': self.hidden_size
        })
        
        # Layer 2: Hidden → Output classes
        node_kwargs = self._build_node_kwargs(
            input_dim=self.node_input_dim,
            current_layer_width=self.hidden_size,
            next_layer_width=None
        )
        
        layer2 = layer_class(
            input_size=self.hidden_size,
            output_size=self.num_classes,
            node_type=node_class,
            node_kwargs=node_kwargs,
            seed=43,
            layer_config=self.layer_config
        )
        self.layers.append(layer2)
        self.layer_sizes.append({
            'type': 'fc',
            'input': self.hidden_size,
            'output': self.num_classes
        })
        
        print(f"Built {len(self.layers)} layers with {self.node_type} nodes")
    
    def _build_node_kwargs(
        self,
        input_dim: int,
        current_layer_width: Optional[int] = None,
        next_layer_width: Optional[int] = None
    ) -> NodeConfig:
        """
        Build node configuration with proper initialization.
        
        Args:
            input_dim: Number of inputs to each node
            current_layer_width: Number of nodes in current layer
            next_layer_width: Number of nodes in next layer
        
        Returns:
            NodeConfig for node initialization
        """
        # Node-specific extra parameters
        extra_params = {}
        
        if self.node_type in ['dwn', 'dwn_stable', 'hybrid']:
            extra_params['use_cuda'] = self.use_cuda
        elif self.node_type == 'probabilistic':
            extra_params['temperature'] = 1.0
            extra_params['eval_mode'] = 'expectation'
            extra_params['use_cuda'] = self.use_cuda
        elif self.node_type == 'fourier':
            extra_params['use_cuda'] = self.use_cuda
            extra_params['use_all_frequencies'] = True
            extra_params['max_amplitude'] = 0.5
        
        return NodeConfig(
            input_dim=input_dim,
            output_dim=1,
            init_fn=None,  # Use default initialization
            init_kwargs=None,
            regularizers=None,
            extra_params=extra_params if extra_params else None
        )
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            x: Input tensor (N, 28, 28) or (N, 784)
        
        Returns:
            Output logits (N, 10)
        """
        batch_size = x.shape[0]
        
        # Flatten
        if x.dim() > 2:
            x = x.view(batch_size, -1)
        
        # Encode
        self.encoder.to(x.device)
        x = self.encoder.encode(x)
        x = torch.clamp(x, 0, 1)
        
        # Through layers
        for layer in self.layers:
            x = layer(x)
        
        # Classification
        x = self.output_layer(x)
        
        return x


class MNISTLinearSmall(MNISTSmallBase):
    """
    MNIST 2-layer fully connected model with linear LUT nodes.
    
    Architecture:
    - Input: 28x28 MNIST images
    - Encoder: Thermometer (4 bits) → 3136 features
    - Layer 1: 3136 → 128 (linear LUT nodes with 4 inputs)
    - Layer 2: 128 → 10 (linear LUT nodes)
    - Output: GroupSum classification
    
    Node type: LinearLUTNode (basic LUT without CUDA extensions)
    Best for: Educational purposes, quick iteration
    """
    
    def __init__(
        self,
        hidden_size: int = 128,
        encoder_bits: int = 4,
        node_input_dim: int = 4,
        flip_probability: float = 0.0,
        grad_stabilization: str = 'none',
        **kwargs
    ):
        """
        Initialize MNIST Linear Small model.
        
        Args:
            hidden_size: Size of hidden layer
            encoder_bits: Thermometer encoder bits
            node_input_dim: Inputs per node
            flip_probability: Bit flip probability for robustness
            grad_stabilization: Gradient stabilization mode
            **kwargs: Additional arguments (unused, for compatibility)
        """
        super().__init__(
            name='mnist_fc_8k_linear',
            node_type='linear_lut',
            node_input_dim=node_input_dim,
            hidden_size=hidden_size,
            encoder_bits=encoder_bits,
            flip_probability=flip_probability,
            grad_stabilization=grad_stabilization,
            use_cuda=False  # LinearLUT doesn't use CUDA
        )


class MNISTDWNSmall(MNISTSmallBase):
    """
    MNIST 2-layer fully connected model with DWN stable nodes.
    
    Architecture: Same as MNISTLinearSmall but using DWN stable nodes.
    
    Node type: DWN Stable (CUDA-accelerated, suitable for FPGA export)
    Best for: FPGA deployment, performance benchmarking
    """
    
    def __init__(
        self,
        hidden_size: int = 128,
        encoder_bits: int = 4,
        node_input_dim: int = 4,
        flip_probability: float = 0.0,
        grad_stabilization: str = 'none',
        use_cuda: bool = True,
        **kwargs
    ):
        """
        Initialize MNIST DWN Small model.
        
        Args:
            hidden_size: Size of hidden layer
            encoder_bits: Thermometer encoder bits
            node_input_dim: Inputs per node
            flip_probability: Bit flip probability for robustness
            grad_stabilization: Gradient stabilization mode
            use_cuda: Whether to use CUDA extensions
            **kwargs: Additional arguments (unused, for compatibility)
        """
        super().__init__(
            name='mnist_fc_8k_dwn',
            node_type='dwn_stable',
            node_input_dim=node_input_dim,
            hidden_size=hidden_size,
            encoder_bits=encoder_bits,
            flip_probability=flip_probability,
            grad_stabilization=grad_stabilization,
            use_cuda=use_cuda
        )
