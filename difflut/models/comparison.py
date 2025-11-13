"""
Utility models showcasing DiffLUT training features.

Models demonstrating specific capabilities:
- BitFlipComparison: 4 variants showing impact of bit flipping
- GradNormComparison: 4 variants showing impact of gradient stabilization
- ResidualInitExample: Showcasing residual initialization
"""

import torch
import torch.nn as nn
from typing import Optional

from difflut import REGISTRY
from difflut.utils import GroupSum
from difflut.nodes.node_config import NodeConfig
from difflut.layers.layer_config import LayerConfig

from .base_model import BaseModel


class ComparisonModelBase(BaseModel):
    """
    Base class for comparison models that showcase different training techniques.
    
    Provides common infrastructure for fair comparisons between variants.
    """
    
    def __init__(
        self,
        name: str,
        node_type: str = 'linear_lut',
        node_input_dim: int = 4,
        hidden_sizes: list = None,
        encoder_bits: int = 4,
        flip_probability: float = 0.0,
        grad_stabilization: str = 'none',
        use_cuda: bool = True,
    ):
        """
        Initialize comparison model.
        
        Args:
            name: Model identifier
            node_type: DiffLUT node type
            node_input_dim: Inputs per node
            hidden_sizes: List of hidden layer widths
            encoder_bits: Thermometer encoder bits
            flip_probability: Bit flip probability for augmentation
            grad_stabilization: Gradient stabilization mode
            use_cuda: Whether to use CUDA extensions
        """
        super().__init__(name=name, input_size=784, num_classes=10)
        
        self.node_type = node_type
        self.node_input_dim = node_input_dim
        self.hidden_sizes = hidden_sizes or [512, 256]
        self.encoder_bits = encoder_bits
        self.use_cuda = use_cuda
        
        # Create encoder
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
        
        # Layers will be built after encoder is fitted
        self.layers = nn.ModuleList()
        self.layer_sizes = []
        
        # Output layer
        self.output_layer = GroupSum(k=self.num_classes, tau=1, use_randperm=False)
    
    def fit_encoder(self, data: torch.Tensor) -> None:
        """Fit encoder on training data."""
        if self.encoder_fitted:
            raise RuntimeError("Encoder already fitted")
        
        batch_size = data.shape[0]
        if data.dim() > 2:
            data_flat = data.view(batch_size, -1)
        else:
            data_flat = data
        
        print(f"Fitting encoder on {len(data_flat)} samples...")
        self.encoder.fit(data_flat)
        
        sample_encoded = self.encoder.encode(data_flat[:1])
        self.encoded_input_size = sample_encoded.shape[1]
        
        self._build_layers()
        self.encoder_fitted = True
    
    def _build_layers(self) -> None:
        """Build network layers."""
        try:
            node_class = REGISTRY.get_node(self.node_type)
            layer_class = REGISTRY.get_layer('random')
        except ValueError as e:
            raise ValueError(f"Failed to get node/layer: {e}")
        
        current_size = self.encoded_input_size
        
        for i, hidden_size in enumerate(self.hidden_sizes):
            next_layer = self.hidden_sizes[i + 1] if i + 1 < len(self.hidden_sizes) else None
            
            node_kwargs = self._build_node_kwargs(
                input_dim=self.node_input_dim,
                current_layer_width=hidden_size,
                next_layer_width=next_layer
            )
            
            layer = layer_class(
                input_size=current_size,
                output_size=hidden_size,
                node_type=node_class,
                node_kwargs=node_kwargs,
                seed=42 + i,
                layer_config=self.layer_config
            )
            
            self.layers.append(layer)
            self.layer_sizes.append({'input': current_size, 'output': hidden_size})
            current_size = hidden_size
        
        # Output layer
        node_kwargs = self._build_node_kwargs(
            input_dim=self.node_input_dim,
            current_layer_width=current_size,
            next_layer_width=None
        )
        
        output_layer = layer_class(
            input_size=current_size,
            output_size=self.num_classes,
            node_type=node_class,
            node_kwargs=node_kwargs,
            seed=100,
            layer_config=self.layer_config
        )
        
        self.layers.append(output_layer)
        self.layer_sizes.append({'input': current_size, 'output': self.num_classes})
    
    def _build_node_kwargs(
        self,
        input_dim: int,
        current_layer_width: Optional[int] = None,
        next_layer_width: Optional[int] = None
    ) -> NodeConfig:
        """Build node configuration."""
        extra_params = {}
        
        if self.node_type in ['dwn', 'dwn_stable', 'hybrid']:
            extra_params['use_cuda'] = self.use_cuda
        elif self.node_type == 'probabilistic':
            extra_params['temperature'] = 1.0
            extra_params['eval_mode'] = 'expectation'
            extra_params['use_cuda'] = self.use_cuda
        
        return NodeConfig(
            input_dim=input_dim,
            output_dim=1,
            init_fn=None,
            init_kwargs=None,
            regularizers=None,
            extra_params=extra_params if extra_params else None
        )
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]
        
        if x.dim() > 2:
            x = x.view(batch_size, -1)
        
        self.encoder.to(x.device)
        x = self.encoder.encode(x)
        x = torch.clamp(x, 0, 1)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_layer(x)
        return x


# ==================== Bit Flipping Comparison Models ====================

class MNISTBitFlipNone(ComparisonModelBase):
    """MNIST model without bit flipping (baseline)."""
    
    def __init__(self):
        super().__init__(
            name='mnist_bitflip_none',
            flip_probability=0.0,
            hidden_sizes=[512, 256]
        )


class MNISTBitFlip5(ComparisonModelBase):
    """MNIST model with 5% bit flipping for robustness."""
    
    def __init__(self):
        super().__init__(
            name='mnist_bitflip_5',
            flip_probability=0.05,
            hidden_sizes=[512, 256]
        )


class MNISTBitFlip10(ComparisonModelBase):
    """MNIST model with 10% bit flipping for robustness."""
    
    def __init__(self):
        super().__init__(
            name='mnist_bitflip_10',
            flip_probability=0.10,
            hidden_sizes=[512, 256]
        )


class MNISTBitFlip20(ComparisonModelBase):
    """MNIST model with 20% bit flipping for maximum robustness."""
    
    def __init__(self):
        super().__init__(
            name='mnist_bitflip_20',
            flip_probability=0.20,
            hidden_sizes=[512, 256]
        )


# ==================== Gradient Normalization Comparison Models ====================

class MNISTGradNormNone(ComparisonModelBase):
    """MNIST model without gradient stabilization (baseline)."""
    
    def __init__(self):
        super().__init__(
            name='mnist_gradnorm_none',
            grad_stabilization='none',
            hidden_sizes=[512, 256]
        )


class MNISTGradNormLayerwise(ComparisonModelBase):
    """MNIST model with layerwise gradient stabilization."""
    
    def __init__(self):
        super().__init__(
            name='mnist_gradnorm_layerwise',
            grad_stabilization='layerwise',
            hidden_sizes=[512, 256]
        )


class MNISTGradNormBatchwise(ComparisonModelBase):
    """MNIST model with batchwise gradient stabilization."""
    
    def __init__(self):
        super().__init__(
            name='mnist_gradnorm_batchwise',
            grad_stabilization='batchwise',
            hidden_sizes=[512, 256]
        )


class MNISTGradNormLayerwiseWithMean(ComparisonModelBase):
    """MNIST model with layerwise gradient stabilization and mean subtraction."""
    
    def __init__(self):
        super().__init__(
            name='mnist_gradnorm_layerwise_mean',
            grad_stabilization='layerwise',
            hidden_sizes=[512, 256]
        )
        # After initialization, enable mean subtraction
        self.layer_config.grad_subtract_mean = True


# ==================== Residual Initialization Showcase ====================

class MNISTResidualInit(BaseModel):
    """
    MNIST model showcasing residual initialization.
    
    Uses residual layers to demonstrate how residual connections
    can affect learning dynamics and initialization.
    """
    
    def __init__(
        self,
        node_type: str = 'linear_lut',
        node_input_dim: int = 4,
        hidden_sizes: list = None,
        intermediate_sizes: list = None,
        encoder_bits: int = 4,
    ):
        """
        Initialize MNIST Residual Init model.
        
        Args:
            node_type: DiffLUT node type
            node_input_dim: Inputs per node
            hidden_sizes: Main hidden layer sizes
            intermediate_sizes: Intermediate residual sizes
            encoder_bits: Thermometer encoder bits
        """
        super().__init__(
            name='mnist_residual_init',
            input_size=784,
            num_classes=10
        )
        
        self.node_type = node_type
        self.node_input_dim = node_input_dim
        self.hidden_sizes = hidden_sizes or [512, 256]
        self.intermediate_sizes = intermediate_sizes or [256, 128]
        self.encoder_bits = encoder_bits
        
        # Create encoder
        from difflut.encoder.thermometer import ThermometerEncoder
        self.encoder = ThermometerEncoder(num_bits=encoder_bits)
        
        # Layer configuration
        self.layer_config = LayerConfig(
            flip_probability=0.0,
            grad_stabilization='none'
        )
        
        # Layers
        self.layers = nn.ModuleList()
        self.layer_sizes = []
        
        # Output layer
        self.output_layer = GroupSum(k=self.num_classes, tau=1, use_randperm=False)
    
    def fit_encoder(self, data: torch.Tensor) -> None:
        """Fit encoder on training data."""
        if self.encoder_fitted:
            raise RuntimeError("Encoder already fitted")
        
        batch_size = data.shape[0]
        if data.dim() > 2:
            data_flat = data.view(batch_size, -1)
        else:
            data_flat = data
        
        print(f"Fitting encoder on {len(data_flat)} samples...")
        self.encoder.fit(data_flat)
        
        sample_encoded = self.encoder.encode(data_flat[:1])
        self.encoded_input_size = sample_encoded.shape[1]
        
        self._build_layers()
        self.encoder_fitted = True
    
    def _build_layers(self) -> None:
        """Build layers with residual configuration."""
        try:
            node_class = REGISTRY.get_node(self.node_type)
            layer_class = REGISTRY.get_layer('residual')
        except ValueError as e:
            # Fall back to random layer if residual not available
            layer_class = REGISTRY.get_layer('random')
        
        current_size = self.encoded_input_size
        
        for i, hidden_size in enumerate(self.hidden_sizes):
            next_layer = self.hidden_sizes[i + 1] if i + 1 < len(self.hidden_sizes) else None
            
            node_kwargs = NodeConfig(
                input_dim=self.node_input_dim,
                output_dim=1,
                init_fn=None,
                init_kwargs=None,
                regularizers=None,
                extra_params=None
            )
            
            # Try to use residual layer if available and intermediate sizes provided
            try:
                if hasattr(layer_class, '__name__') and 'Residual' in layer_class.__name__:
                    layer = layer_class(
                        input_size=current_size,
                        output_size=hidden_size,
                        intermediate_sizes=self.intermediate_sizes,
                        node_type=node_class,
                        node_kwargs=node_kwargs,
                        seed=42 + i,
                        layer_config=self.layer_config
                    )
                else:
                    raise AttributeError
            except (AttributeError, TypeError):
                # Fall back to regular layer
                layer = layer_class(
                    input_size=current_size,
                    output_size=hidden_size,
                    node_type=node_class,
                    node_kwargs=node_kwargs,
                    seed=42 + i,
                    layer_config=self.layer_config
                )
            
            self.layers.append(layer)
            self.layer_sizes.append({'input': current_size, 'output': hidden_size})
            current_size = hidden_size
        
        # Output layer
        node_kwargs = NodeConfig(
            input_dim=self.node_input_dim,
            output_dim=1,
            init_fn=None,
            init_kwargs=None,
            regularizers=None,
            extra_params=None
        )
        
        output_layer = layer_class(
            input_size=current_size,
            output_size=self.num_classes,
            node_type=node_class,
            node_kwargs=node_kwargs,
            seed=100,
            layer_config=self.layer_config
        )
        
        self.layers.append(output_layer)
        self.layer_sizes.append({'input': current_size, 'output': self.num_classes})
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]
        
        if x.dim() > 2:
            x = x.view(batch_size, -1)
        
        self.encoder.to(x.device)
        x = self.encoder.encode(x)
        x = torch.clamp(x, 0, 1)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_layer(x)
        return x
