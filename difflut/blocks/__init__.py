"""
Blocks module for DiffLUT.

Blocks are composite modules that consist of multiple layers but are not complete models.
Examples: ConvolutionalBlock, ResidualBlock, etc.

Unlike layers which are individual processing units, blocks combine multiple layers
to create reusable architectural patterns.
"""

from .convolutional import ConvolutionalLayer, ConvolutionConfig

__all__ = ["ConvolutionConfig", "ConvolutionalLayer"]
