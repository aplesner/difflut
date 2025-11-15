"""
Blocks module for DiffLUT.

Blocks are composite modules that consist of multiple layers but are not complete models.
Examples: ConvolutionalBlock, ResidualBlock, AttentionBlock, etc.

Unlike layers which are individual processing units, blocks combine multiple layers
to create reusable architectural patterns.

Core Components:
- BlockConfig: Configuration dataclass for block parameters
- BaseLUTBlock: Base class for all DiffLUT blocks
- ConvolutionalLayer: Tree-based convolutional block implementation
"""

from .base_block import BaseLUTBlock
from .block_config import BlockConfig
from .convolutional import ConvolutionalLayer

__all__ = [
    "BlockConfig",
    "BaseLUTBlock",
    "ConvolutionalLayer",
]
