"""
Base block infrastructure for DiffLUT blocks.

Provides a template for all DiffLUT blocks with config management
and standardized structure. Blocks are composable modules that consist
of layers but are not complete models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .block_config import BlockConfig


class BaseLUTBlock(ABC, nn.Module):
    """
    Base class for all DiffLUT blocks.

    Blocks are composable modules consisting of one or more layers but not
    complete models. They are used as building blocks within larger models.

    Examples:
        - ConvolutionalBlock: Tree-based convolutional processing
        - ResidualBlock: Skip connection + residual computation
        - AttentionBlock: Multi-head attention mechanism
        - PoolingBlock: Spatial or feature pooling
        - NormalizationBlock: Batch/Layer normalization

    Features:
        - Config-based initialization for reproducibility
        - Runtime parameter overrides (safe to change without affecting structure)
        - Hooks for subclasses to respond to runtime changes
        - Standardized serialization and deserialization
        - Proper integration with model hierarchy

    Subclasses should implement:
        - forward(): The actual forward pass logic
        - on_runtime_update(): React to runtime parameter changes (optional)
        - get_output_shape(): Calculate output shape given input shape (optional)

    Example:
        ```python
        # Define a custom block
        class MyCustomBlock(BaseLUTBlock):
            def __init__(self, config: BlockConfig):
                super().__init__(config)
                # Initialize your layers here
                self.layer1 = SomeLayer(...)
                self.layer2 = AnotherLayer(...)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Implement forward pass
                x = self.layer1(x)
                x = self.layer2(x)
                return x

            def on_runtime_update(self):
                # Optional: respond to runtime parameter changes
                if 'dropout_rate' in self.runtime:
                    self.layer1.dropout_rate = self.runtime['dropout_rate']

        # Use the block
        config = BlockConfig(
            block_type='my_custom_block',
            seed=42,
            # ... other parameters
        )
        block = MyCustomBlock(config)
        ```
    """

    def __init__(self, config: BlockConfig):
        """
        Initialize the base block.

        Args:
            config: BlockConfig instance containing all block parameters

        Raises:
            ValueError: If config is invalid or missing required parameters
        """
        super().__init__()

        # Validate config
        if not isinstance(config, BlockConfig):
            raise TypeError(
                f"config must be a BlockConfig instance, got {type(config).__name__}"
            )

        self.config = config

        # Copy runtime parameters (can be updated without affecting structure)
        self.runtime: Dict[str, Any] = {}

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block.

        Args:
            x: Input tensor

        Returns:
            Output tensor

        Raises:
            NotImplementedError: Subclasses must implement this
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def apply_runtime_overrides(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Apply runtime-only parameter overrides.

        These are parameters that can be changed without affecting the block
        structure (e.g., dropout rate, bit flip probability, temperature, eval mode).

        Args:
            overrides: Dictionary of runtime parameters to override

        Example:
            ```python
            block.apply_runtime_overrides({
                'flip_probability': 0.05,
                'temperature': 0.8
            })
            ```
        """
        if overrides:
            self.runtime.update(overrides)
            self.on_runtime_update()

    def on_runtime_update(self) -> None:
        """
        Hook for subclasses to react to runtime parameter changes.

        This is called after runtime parameters are updated via
        apply_runtime_overrides(). Subclasses can override this to
        apply the changes to internal components.

        Example:
            ```python
            def on_runtime_update(self):
                if 'flip_probability' in self.runtime:
                    flip_prob = self.runtime['flip_probability']
                    for layer in self.layers:
                        layer.flip_probability = flip_prob

                if 'eval_mode' in self.runtime:
                    eval_mode = self.runtime['eval_mode']
                    for node in self.nodes:
                        node.eval_mode = eval_mode
            ```
        """
        pass  # Default: no-op, subclasses can override

    def get_output_shape(self, input_shape: torch.Size) -> Optional[torch.Size]:
        """
        Calculate output shape given input shape.

        This is an optional method for blocks that can statically determine
        their output shape. Useful for model building and validation.

        Args:
            input_shape: Shape of input tensor (excluding batch dimension)
                Example: torch.Size([3, 32, 32]) for 3x32x32 images

        Returns:
            Output shape (excluding batch dimension), or None if not determinable

        Example:
            ```python
            block = ConvolutionalBlock(config)
            input_shape = torch.Size([3, 32, 32])  # 3 channels, 32x32 spatial

            output_shape = block.get_output_shape(input_shape)
            # Returns torch.Size([32, 30, 30]) for 32 channels, 30x30 spatial
            # (considering receptive field reduction)
            ```
        """
        return None  # Default: not determinable

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the block's configuration.

        Returns:
            Dictionary with block configuration details

        Example:
            ```python
            summary = block.get_config_summary()
            print(summary)
            # {
            #     'block_type': 'convolutional',
            #     'seed': 42,
            #     'in_channels': 3,
            #     'out_channels': 32,
            #     'receptive_field': (5, 5),
            #     ...
            # }
            ```
        """
        return {
            "block_type": self.config.block_type,
            "seed": self.config.seed,
        }

    def __repr__(self) -> str:
        """Return detailed representation of the block."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  block_type={self.config.block_type}\n"
            f"  seed={self.config.seed}\n"
            f")"
        )
