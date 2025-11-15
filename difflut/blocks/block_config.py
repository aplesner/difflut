"""
Configuration class for block parameters.

This module provides a typed, maintainable way to specify block configuration
following the same pattern as NodeConfig, LayerConfig, and ModelConfig.

Blocks are composable modules that consist of layers but are not complete models.
Examples: ConvolutionalBlock, ResidualBlock, AttentionBlock, etc.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple, Union


@dataclass
class BlockConfig:
    """
    Flexible configuration for DiffLUT blocks.

    This config is block-type-agnostic and stores only the parameters relevant
    to each specific block. Different block types may have completely different
    architectures (convolutional, residual, attention, etc.) and this config
    adapts to each.

    Structural Parameters (must match pretrained weights):
        - block_type: Type of block class to instantiate (e.g., "convolutional")
        - seed: Random seed for reproducibility
        - All other fields store block-specific architectural parameters

    Design Philosophy:
        - Store only parameters actually used by the specific block
        - No null/unused fields
        - Block classes define what parameters they need, not this config
        - Support heterogeneous block architectures (convolutional, residual,
          attention, pooling, normalization, etc.)

    Common Block Parameters (Convolutional):
        tree_depth: int, Depth of tree architecture (default: 2)
        in_channels: int, Number of input channels
        out_channels: int, Number of output channels
        receptive_field: Tuple[int, int], Size of receptive field (default: (5, 5))
        stride: Tuple[int, int], Stride for convolution (default: (1, 1))
        padding: Tuple[int, int], Padding for convolution (default: (0, 0))
        patch_chunk_size: Optional[int], Process patches in chunks to save memory

    Layer and Node Configuration:
        node_type: str, Name of node type to use (e.g., "dwn", "fourier")
        node_kwargs: Dict, Node configuration parameters
        layer_type: str, Name of layer type to use (e.g., "random")
        layer_config: Dict, Layer training configuration (bit flip, grad stabilization)

    Training Parameters:
        flip_probability: float, Probability of bit flipping during training [0, 1]
        grad_stabilization: str, Gradient stabilization mode ('none', 'layerwise', 'batchwise')
        grad_target_std: float, Target standard deviation for gradient rescaling
        grad_subtract_mean: bool, Whether to subtract mean during gradient stabilization
        grad_epsilon: float, Epsilon for numerical stability

    Connection Strategy:
        grouped_connections: bool, Use channel-aware connections (default: False)
        ensure_full_coverage: bool, Ensure each input is used at least once (default: False)

    Example (Convolutional):
        ```python
        # Create configuration for convolutional block
        block_config = BlockConfig(
            block_type='convolutional',
            seed=42,
            tree_depth=2,
            in_channels=3,
            out_channels=32,
            receptive_field=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
            node_type='dwn',
            layer_type='random',
            flip_probability=0.1,
            grad_stabilization='layerwise'
        )

        # Or pass as dictionaries for node/layer specifics
        block_config = BlockConfig(
            block_type='convolutional',
            seed=42,
            in_channels=3,
            out_channels=32,
            node_kwargs={
                'input_dim': 6,
                'output_dim': 1,
            },
            layer_config={
                'flip_probability': 0.1,
                'grad_stabilization': 'layerwise'
            }
        )
        ```
    """

    # ==================== Core Parameters ====================
    block_type: str  # e.g., "convolutional", "residual", "attention"
    seed: int = 42

    # ==================== Block-Specific Structural Parameters ====================
    # Common convolutional block parameters
    tree_depth: Optional[int] = None
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    receptive_field: Optional[Tuple[int, int]] = None
    stride: Optional[Tuple[int, int]] = None
    padding: Optional[Tuple[int, int]] = None
    patch_chunk_size: Optional[int] = None

    # ==================== Node and Layer Configuration ====================
    node_type: Optional[str] = None  # Name of registered node type
    layer_type: Optional[str] = None  # Name of registered layer type
    n_inputs_per_node: Optional[int] = None  # Number of inputs per node

    # Flexible dictionaries for detailed node/layer configuration
    node_kwargs: Dict[str, Any] = field(default_factory=dict)
    layer_config: Dict[str, Any] = field(default_factory=dict)

    # ==================== Training Parameters ====================
    # Bit flip augmentation
    flip_probability: float = 0.0

    # Gradient stabilization
    grad_stabilization: str = "none"
    grad_target_std: float = 1.0
    grad_subtract_mean: bool = False
    grad_epsilon: float = 1e-8

    # ==================== Connection Strategy ====================
    grouped_connections: bool = False
    ensure_full_coverage: bool = False

    # ==================== Flexible Storage ====================
    # Any additional block-specific parameters go here
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate block_type
        if not isinstance(self.block_type, str) or not self.block_type.strip():
            raise ValueError(
                f"block_type must be a non-empty string, got {self.block_type}. "
                f"Example: block_type='convolutional'"
            )

        # Validate seed
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(
                f"seed must be a non-negative integer, got {self.seed}. " f"Example: seed=42"
            )

        # Validate flip_probability
        if not isinstance(self.flip_probability, (int, float)) or not (
            0.0 <= self.flip_probability <= 1.0
        ):
            raise ValueError(
                f"flip_probability must be a float in [0, 1], got {self.flip_probability}. "
                f"Example: flip_probability=0.1 for 10% bit flipping during training."
            )

        # Validate grad_stabilization
        valid_grad_modes = ["none", "layerwise", "batchwise"]
        if self.grad_stabilization not in valid_grad_modes:
            raise ValueError(
                f"grad_stabilization must be one of {valid_grad_modes}, got '{self.grad_stabilization}'. "
                f"'layerwise': normalize per layer, 'batchwise': normalize per batch sample, 'none': disabled"
            )

        # Validate grad_target_std
        if not isinstance(self.grad_target_std, (int, float)) or self.grad_target_std <= 0:
            raise ValueError(
                f"grad_target_std must be a positive number, got {self.grad_target_std}. "
                f"Example: grad_target_std=1.0 for unit variance"
            )

        # Validate grad_epsilon
        if not isinstance(self.grad_epsilon, (int, float)) or self.grad_epsilon <= 0:
            raise ValueError(
                f"grad_epsilon must be a positive number, got {self.grad_epsilon}. "
                f"Example: grad_epsilon=1e-8"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "BlockConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary with configuration parameters

        Returns:
            BlockConfig instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        if "block_type" not in data:
            raise ValueError("block_type is required in configuration dictionary")

        return BlockConfig(**data)
