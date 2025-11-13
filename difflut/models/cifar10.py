"""
CIFAR-10 model zoo implementations.

Models for CIFAR-10 classification with convolutional layers highlighting
DiffLUT capabilities on color images:
- CIFAR10Conv: 2 conv layers with max pooling + 2 FC layers
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


class CIFAR10Conv(BaseModel):
    """
    CIFAR-10 convolutional model showcasing thermometer encoding on color images.
    
    Architecture:
    - Input: 32x32 RGB images (3 channels)
    - Encoder: Thermometer encoding (4 bits) → 384 features (3*32*32*4)
    - Conv Layer 1: Input 384 → Output 128 features (2D convolution)
    - Max Pooling: 2x2 stride
    - Conv Layer 2: Input features → Output 256 features (2D convolution)
    - Max Pooling: 2x2 stride
    - FC Layer 1: Flattened features → 256 nodes
    - FC Layer 2: 256 nodes → 10 classes
    - Classifier: GroupSum for 10 classes
    
    Design choices:
    - Thermometer encoding on RGB: Shows how encoding works with color images
    - 2 conv layers with pooling: Demonstrates spatial feature extraction
    - 2 FC layers: Balance between capacity and model size
    - Suitable for end-to-end training on CIFAR-10
    
    This model demonstrates:
    - Multi-channel input handling
    - Convolutional LUT processing
    - Integration of spatial and fully-connected layers
    """
    
    def __init__(
        self,
        node_type: str = 'linear_lut',
        node_input_dim: int = 4,
        conv_node_input_dim: int = 6,
        encoder_bits: int = 4,
        flip_probability: float = 0.0,
        grad_stabilization: str = 'none',
        use_cuda: bool = True,
    ):
        """
        Initialize CIFAR-10 Convolutional model.
        
        Args:
            node_type: Type of DiffLUT node for FC layers
            node_input_dim: Inputs per FC node (typically 4 or 6)
            conv_node_input_dim: Inputs per conv node (typically 6)
            encoder_bits: Number of thermometer encoder bits per channel
            flip_probability: Bit flip probability for robustness [0, 1]
            grad_stabilization: Gradient stabilization mode ('none', 'layerwise')
            use_cuda: Whether to use CUDA extensions
        """
        super().__init__(
            name='cifar10_conv',
            input_size=32*32*3,  # 3072 pixels (RGB)
            num_classes=10
        )
        
        # Model configuration
        self.node_type = node_type
        self.node_input_dim = node_input_dim
        self.conv_node_input_dim = conv_node_input_dim
        self.encoder_bits = encoder_bits
        self.use_cuda = use_cuda
        
        # Create encoder (will be fitted on training data)
        from difflut.encoder.thermometer import ThermometerEncoder
        self.encoder = ThermometerEncoder(num_bits=encoder_bits)
        
        # Layer configuration
        self.layer_config = LayerConfig(
            flip_probability=flip_probability,
            grad_stabilization=grad_stabilization,
            grad_target_std=1.0,
            grad_subtract_mean=False,
            grad_epsilon=1e-8
        )
        
        # Convolutional layer configuration
        self.conv_layer_config = LayerConfig(
            flip_probability=flip_probability,
            grad_stabilization=grad_stabilization,
            grad_target_std=1.0,
            grad_subtract_mean=False,
            grad_epsilon=1e-8
        )
        
        # Layers will be built after encoder is fitted
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.layer_sizes = []
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Output layer
        self.output_layer = GroupSum(k=self.num_classes, tau=1, use_randperm=False)
    
    def fit_encoder(self, data: torch.Tensor) -> None:
        """
        Fit thermometer encoder on training data.
        
        Args:
            data: Training data tensor (N, 3, 32, 32) - CIFAR-10 images
        
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
        self.encoded_channels = sample_encoded.shape[1] // (32 * 32)
        self.encoded_input_size = sample_encoded.shape[1]
        print(f"Encoded input size: {self.encoded_input_size} ({self.encoded_channels} channels per pixel)")
        
        # Build layers
        self._build_layers()
        self.encoder_fitted = True
    
    def _build_layers(self) -> None:
        """
        Build network layers after encoder is fitted.
        
        Creates:
        - Conv Layer 1: encoded_size (H, W) → 128 features
        - Max Pooling: 2x2
        - Conv Layer 2: 128 features → 256 features
        - Max Pooling: 2x2
        - FC Layer 1: flattened → 256 nodes
        - FC Layer 2: 256 nodes → 10 classes
        """
        try:
            node_class = REGISTRY.get_node(self.node_type)
            conv_layer_class = REGISTRY.get_convolutional_layer('convolutional')
            layer_class = REGISTRY.get_layer('random')  # For FC layers
        except ValueError as e:
            raise ValueError(f"Failed to get node/layer from registry: {e}")
        
        # Convolutional layer structure:
        # Input: (batch, 3*4, 32, 32) = (batch, 12, 32, 32) with encoded channels
        # Conv1 Output: (batch, 128, 32, 32)
        # Pool1: (batch, 128, 16, 16)
        # Conv2 Output: (batch, 256, 16, 16)
        # Pool2: (batch, 256, 8, 8)
        # Flatten: (batch, 256*8*8) = (batch, 16384)
        
        # Build convolutional nodes
        conv_node_kwargs = self._build_node_kwargs(
            input_dim=self.conv_node_input_dim,
            node_type='conv'
        )
        
        # Conv Layer 1: Input encoded channels → 128 output features
        conv1 = conv_layer_class(
            input_size=self.encoded_channels,  # Encoded channels per pixel
            output_size=128,
            kernel_size=(3, 3),
            input_height=32,
            input_width=32,
            node_type=node_class,
            node_kwargs=conv_node_kwargs,
            seed=42,
            layer_config=self.conv_layer_config
        )
        self.conv_layers.append(conv1)
        self.layer_sizes.append({
            'type': 'conv',
            'input': f'{self.encoded_channels}@32x32',
            'output': '128@32x32'
        })
        
        # Conv Layer 2: 128 → 256 output features
        conv_node_kwargs = self._build_node_kwargs(
            input_dim=self.conv_node_input_dim,
            node_type='conv'
        )
        
        conv2 = conv_layer_class(
            input_size=128,
            output_size=256,
            kernel_size=(3, 3),
            input_height=16,  # After first pooling
            input_width=16,
            node_type=node_class,
            node_kwargs=conv_node_kwargs,
            seed=43,
            layer_config=self.conv_layer_config
        )
        self.conv_layers.append(conv2)
        self.layer_sizes.append({
            'type': 'conv',
            'input': '128@16x16',
            'output': '256@16x16'
        })
        
        # After pooling: (batch, 256, 8, 8) = (batch, 16384) when flattened
        fc_input_size = 256 * 8 * 8
        
        # FC Layer 1: Flattened conv → 256 nodes
        fc_node_kwargs = self._build_node_kwargs(
            input_dim=self.node_input_dim,
            current_layer_width=None,
            next_layer_width=256,
            node_type='fc'
        )
        
        fc1 = layer_class(
            input_size=fc_input_size,
            output_size=256,
            node_type=node_class,
            node_kwargs=fc_node_kwargs,
            seed=44,
            layer_config=self.layer_config
        )
        self.fc_layers.append(fc1)
        self.layer_sizes.append({
            'type': 'fc',
            'input': fc_input_size,
            'output': 256
        })
        
        # FC Layer 2: 256 → 10 classes
        fc_node_kwargs = self._build_node_kwargs(
            input_dim=self.node_input_dim,
            current_layer_width=256,
            next_layer_width=None,
            node_type='fc'
        )
        
        fc2 = layer_class(
            input_size=256,
            output_size=10,
            node_type=node_class,
            node_kwargs=fc_node_kwargs,
            seed=45,
            layer_config=self.layer_config
        )
        self.fc_layers.append(fc2)
        self.layer_sizes.append({
            'type': 'fc',
            'input': 256,
            'output': 10
        })
        
        print(f"Built {len(self.conv_layers)} conv layers and {len(self.fc_layers)} FC layers")
    
    def _build_node_kwargs(
        self,
        input_dim: int,
        node_type: str = 'fc',
        current_layer_width: Optional[int] = None,
        next_layer_width: Optional[int] = None
    ) -> NodeConfig:
        """
        Build node configuration.
        
        Args:
            input_dim: Number of inputs to each node
            node_type: 'conv' or 'fc'
            current_layer_width: Number of nodes in current layer
            next_layer_width: Number of nodes in next layer
        
        Returns:
            NodeConfig for node initialization
        """
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
            x: Input tensor (N, 3, 32, 32) CIFAR-10 images
        
        Returns:
            Output logits (N, 10)
        """
        batch_size = x.shape[0]
        
        # Flatten for encoding
        if x.dim() == 4:
            x_flat = x.view(batch_size, -1)
        else:
            x_flat = x
        
        # Encode
        self.encoder.to(x.device)
        x = self.encoder.encode(x_flat)
        x = torch.clamp(x, 0, 1)
        
        # Reshape to (batch, encoded_channels, 32, 32) for convolution
        x = x.view(batch_size, self.encoded_channels, 32, 32)
        
        # Conv Layer 1 + Pool
        x = self.conv_layers[0](x)
        x = self.pool(x)
        
        # Conv Layer 2 + Pool
        x = self.conv_layers[1](x)
        x = self.pool(x)
        
        # Flatten for FC layers
        x = x.view(batch_size, -1)
        
        # FC Layer 1
        x = self.fc_layers[0](x)
        
        # FC Layer 2
        x = self.fc_layers[1](x)
        
        # Classification
        x = self.output_layer(x)
        
        return x
