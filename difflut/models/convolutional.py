"""
SimpleConvolutional model for DiffLUT.

A config-based convolutional network implementation using LUT-based blocks.
Designed for processing image data with spatial structure preservation.

Similar to SimpleFeedForward but uses ConvolutionalBlock instead of dense layers.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..blocks.convolutional import ConvolutionConfig, ConvolutionalLayer
from ..layers.layer_config import LayerConfig
from ..nodes.node_config import NodeConfig
from ..registry import REGISTRY
from ..utils import GroupSum
from .base_model import BaseLUTModel
from .model_config import ModelConfig


class SimpleConvolutional(BaseLUTModel):
    """
    Simple convolutional network using DiffLUT nodes.

    This model:
    - Uses config-based initialization for reproducibility
    - Supports pretrained weights via ModelConfig
    - Handles runtime parameter overrides
    - Automatically fits encoder on training data
    - Manages node initialization and regularization via registry
    - Processes spatial data with convolutional blocks

    Architecture:
    - Encoder for preprocessing
    - Series of ConvolutionalBlock layers with progressive feature maps
    - Adaptive global average pooling to flatten
    - Final output layer (GroupSum)

    Differences from SimpleFeedForward:
    - Designed for image/spatial data instead of flat features
    - Uses ConvolutionalBlocks instead of dense layers
    - Maintains spatial structure through the network
    - Adaptive pooling at the end for flexible input sizes
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the SimpleConvolutional model.

        Args:
            config: ModelConfig instance with all model parameters
                   Should include:
                   - input_size: (C, H, W) or just number of channels
                   - conv_layer_widths: List of output channels for each conv layer
                   - conv_kernel_size: Kernel size for convolutions
                   - conv_stride: Stride for convolutions
                   - conv_padding: Padding for convolutions
                   - num_classes: Number of output classes
        """
        super().__init__(config)

        # Setup encoder
        encoder_config = config.encoder_config
        encoder_name = encoder_config.get("name", "thermometer")
        encoder_params = {k: v for k, v in encoder_config.items() if k != "name"}

        encoder_class = REGISTRY.get_encoder(encoder_name)
        self.encoder = encoder_class(**encoder_params)
        self.encoder_fitted = False

        # Input/output specifications
        self.input_size = config.input_size  # (C, H, W) or just C
        self.num_classes = config.num_classes

        # Extract input channels
        if isinstance(self.input_size, (tuple, list)):
            self.in_channels = self.input_size[0]
        else:
            self.in_channels = self.input_size

        # Convolutional layers will be built after encoder is fitted
        self.conv_layers = nn.ModuleList()
        self.encoded_input_size = None

        # Convolutional parameters from config
        self.conv_kernel_size = config.runtime.get("conv_kernel_size", 5)
        self.conv_stride = config.runtime.get("conv_stride", 1)
        self.conv_padding = config.runtime.get("conv_padding", 0)

        # Output layer (GroupSum)
        output_tau = config.runtime.get("output_tau", 1.0)
        self.output_layer = GroupSum(k=self.num_classes, tau=output_tau, use_randperm=False)

    def fit_encoder(self, data: torch.Tensor):
        """
        Fit the encoder on training data and build convolutional layers.

        Args:
            data: Training data tensor (N, C, H, W) for images
                  Will be flattened for encoder fitting
        """
        if self.encoder_fitted:
            print("Warning: Encoder already fitted, skipping...")
            return

        # Flatten data for encoder fitting
        batch_size = data.shape[0]
        data_flat = data.view(batch_size, -1)

        # Update input channels if needed
        if data.dim() >= 3:
            actual_channels = data.shape[1]
            if self.in_channels != actual_channels:
                print(
                    f"Warning: Input channels mismatch. "
                    f"Config: {self.in_channels}, Data: {actual_channels}"
                )
                self.in_channels = actual_channels

        # Fit encoder
        print(f"Fitting encoder on {len(data_flat)} samples with shape {data_flat.shape}...")
        self.encoder.fit(data_flat)

        # Get encoded size by encoding a sample
        sample_encoded = self.encoder.encode(data_flat[:1])
        self.encoded_input_size = sample_encoded.shape[1]
        print(f"Encoded input size: {self.encoded_input_size}")

        # Now build convolutional layers
        self._build_conv_layers()

        self.encoder_fitted = True

    def _build_conv_layers(self):
        """
        Build convolutional layers after encoder is fitted.

        Uses the registry to get layer and node classes, properly handles
        initialization and regularization.
        """
        config = self.config
        runtime = self.runtime

        # Get layer and node classes from registry
        layer_class = REGISTRY.get_layer(config.layer_type)
        node_class = REGISTRY.get_node(config.node_type)

        # Create LayerConfig for layer-level parameters
        layer_cfg = LayerConfig(
            flip_probability=runtime.get("flip_probability", 0.0),
            grad_stabilization=runtime.get("grad_stabilization", "none"),
            grad_target_std=runtime.get("grad_target_std", 1.0),
            grad_subtract_mean=runtime.get("grad_subtract_mean", False),
            grad_epsilon=runtime.get("grad_epsilon", 1e-8),
        )

        # Get layer widths (output channels for each conv layer)
        conv_layer_widths = config.runtime.get("conv_layer_widths", [32, 64, 64])

        # Get tree depth for convolutional blocks
        tree_depth = config.runtime.get("tree_depth", 2)

        # Build each convolutional layer
        in_channels = self.in_channels
        for layer_idx, out_channels in enumerate(conv_layer_widths):
            # Build node configuration
            fan_in = config.node_input_dim

            # For conv layers, fan_out is harder to calculate precisely
            # Use a reasonable heuristic
            receptive_field = self.conv_kernel_size
            receptive_area = receptive_field * receptive_field
            fan_out = (out_channels * config.node_input_dim) / (in_channels * receptive_area)

            node_kwargs = self._build_node_config(fan_in=fan_in, fan_out=fan_out)

            # Create ConvolutionConfig
            conv_config = ConvolutionConfig(
                tree_depth=tree_depth,
                in_channels=in_channels,
                out_channels=out_channels,
                receptive_field=self.conv_kernel_size,
                stride=self.conv_stride,
                padding=self.conv_padding,
                seed=config.seed + layer_idx,  # Different seed for each layer
            )

            # Create convolutional layer
            conv_layer = ConvolutionalLayer(
                convolution_config=conv_config,
                node_type=node_class,
                node_kwargs=node_kwargs,
                layer_type=layer_class,
                n_inputs_per_node=config.node_input_dim,
                layer_config=layer_cfg,
                grouped_connections=runtime.get("grouped_connections", False),
                ensure_full_coverage=runtime.get("ensure_full_coverage", False),
            )

            self.conv_layers.append(conv_layer)
            in_channels = out_channels

        print(f"Built {len(self.conv_layers)} convolutional layers with widths {conv_layer_widths}")

    def _build_node_config(self, fan_in: int, fan_out: float) -> NodeConfig:
        """
        Build node configuration with proper initialization and regularization.

        Args:
            fan_in: Number of inputs to each node
            fan_out: Average number of outputs each node connects to

        Returns:
            NodeConfig instance
        """
        config = self.config
        runtime = self.runtime

        # Initialize common parameters
        init_fn = None
        init_kwargs = {}
        regularizers = {}
        extra_params = {}

        # Handle initializer from runtime config
        init_fn_name = runtime.get("init_fn", None)
        if init_fn_name:
            try:
                init_fn = REGISTRY.get_initializer(init_fn_name)

                # Build init_kwargs based on initializer signature
                import inspect

                sig = inspect.signature(init_fn)

                # Add fan_in/fan_out if needed
                if "fan_in" in sig.parameters:
                    init_kwargs["fan_in"] = fan_in
                if "fan_out" in sig.parameters:
                    init_kwargs["fan_out"] = fan_out

                # Add other init parameters from runtime
                for param_name in sig.parameters:
                    if param_name not in ["node", "fan_in", "fan_out", "kwargs"]:
                        runtime_key = f"init_{param_name}"
                        if runtime_key in runtime:
                            init_kwargs[param_name] = runtime[runtime_key]

            except ValueError as e:
                print(f"Warning: Could not load initializer '{init_fn_name}': {e}")

        # Handle regularizers from runtime config
        regularizers_config = runtime.get("regularizers", {})
        if regularizers_config:
            for reg_name, reg_config in regularizers_config.items():
                try:
                    reg_fn = REGISTRY.get_regularizer(reg_name)
                    weight = (
                        reg_config.get("weight", 1.0)
                        if isinstance(reg_config, dict)
                        else reg_config
                    )
                    regularizers[reg_name] = {"fn": reg_fn, "weight": weight}
                except ValueError as e:
                    print(f"Warning: Could not load regularizer '{reg_name}': {e}")

        # Add node-specific parameters from runtime
        node_type = config.node_type

        if node_type in ["dwn", "hybrid", "dwn_stable"]:
            extra_params["use_cuda"] = runtime.get("use_cuda", True)

        elif node_type == "probabilistic":
            extra_params["temperature"] = runtime.get("temperature", 1.0)
            extra_params["eval_mode"] = runtime.get("eval_mode", "expectation")
            extra_params["use_cuda"] = runtime.get("use_cuda", True)

        elif node_type == "fourier":
            extra_params["use_cuda"] = runtime.get("use_cuda", True)
            extra_params["use_all_frequencies"] = runtime.get("use_all_frequencies", True)
            extra_params["max_amplitude"] = runtime.get("max_amplitude", 0.5)

        elif node_type == "neurallut":
            extra_params["hidden_width"] = runtime.get("hidden_width", 8)
            extra_params["depth"] = runtime.get("depth", 2)
            extra_params["skip_interval"] = runtime.get("skip_interval", 2)
            extra_params["activation"] = runtime.get("activation", "relu")
            extra_params["tau_start"] = runtime.get("tau_start", 1.0)
            extra_params["tau_min"] = runtime.get("tau_min", 0.0001)
            extra_params["tau_decay_iters"] = runtime.get("tau_decay_iters", 1000.0)
            extra_params["ste"] = runtime.get("ste", False)
            extra_params["grad_factor"] = runtime.get("grad_factor", 1.0)

        elif node_type == "polylut":
            extra_params["degree"] = runtime.get("degree", 2)

        # Create and return NodeConfig
        return NodeConfig(
            input_dim=config.node_input_dim,
            output_dim=1,  # Always 1 for standard nodes
            init_fn=init_fn,
            init_kwargs=init_kwargs if init_kwargs else None,
            regularizers=regularizers if regularizers else None,
            extra_params=extra_params,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the entire convolutional network.

        Args:
            x: Input tensor (N, C, H, W) - raw images
               Will be encoded and passed through convolutional DiffLUT blocks

        Returns:
            Output logits (N, num_classes)
        """
        if not self.encoder_fitted:
            raise RuntimeError(
                "Encoder must be fitted before forward pass. Call fit_encoder() first."
            )

        # Ensure input is 4D (batch, channels, height, width)
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (N, C, H, W), got shape {x.shape}. "
                "SimpleConvolutional expects image data."
            )

        batch_size = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]

        # Move encoder to same device as input
        self.encoder.to(x.device)

        # Encode input: (N, C, H, W) -> (N, C*H*W) -> (N, encoded_size)
        x_flat = x.view(batch_size, -1)
        x_encoded = self.encoder.encode(x_flat)

        # Clamp to valid range [0, 1]
        x_encoded = torch.clamp(x_encoded, 0, 1)

        # Reshape encoded features to spatial layout for convolutional processing
        # encoded_size features -> (N, encoded_size, 1, 1) spatial layout
        # This is treated as encoded_size "channels" at a single spatial location
        # The convolutional layer will extract patches from this
        # To ensure we have enough spatial dimensions for the kernel, we reshape to
        # a square spatial layout that's large enough
        # e.g., if encoded_size=768, we can reshape to (N, c, h, w) where c*h*w = 768
        
        # Calculate a reasonable spatial layout: try to make it roughly square-ish
        # For now, use a simple approach: create a spatial layout with enough padding
        # Reshape to (batch, channels, spatial_h, spatial_w) where channels*spatial_h*spatial_w = encoded_size
        # Use channels = in_channels (reuse the original channel count as a heuristic)
        
        encoded_size = x_encoded.shape[1]
        # Create spatial dimensions to accommodate kernel size without running out of space
        # Minimum spatial size should be > kernel_size to allow at least 1 patch
        min_spatial = self.conv_kernel_size + 2  # Some padding
        
        # Try to create a square-ish layout
        spatial_pixels = (min_spatial + encoded_size // self.in_channels) 
        spatial_h = spatial_w = int(spatial_pixels ** 0.5) + 1
        
        # Adjust if necessary to fit encoded_size
        while spatial_h * spatial_w * self.in_channels < encoded_size:
            spatial_h += 1
        
        # Reshape: (N, encoded_size) -> (N, in_channels, spatial_h, spatial_w)
        # Pad if needed
        if spatial_h * spatial_w * self.in_channels > encoded_size:
            x_encoded_padded = torch.zeros(
                batch_size,
                spatial_h * spatial_w * self.in_channels,
                device=x_encoded.device,
                dtype=x_encoded.dtype,
            )
            x_encoded_padded[:, :encoded_size] = x_encoded
            x = x_encoded_padded.view(batch_size, self.in_channels, spatial_h, spatial_w)
        else:
            x = x_encoded.view(batch_size, self.in_channels, spatial_h, spatial_w)

        # Pass through convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Global average pooling to reduce spatial dimensions
        # (N, C, H, W) -> (N, C)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(batch_size, -1)

        # Final classification layer (GroupSum)
        x = self.output_layer(x)

        return x

    def on_runtime_update(self):
        """
        Apply runtime parameter changes to internal components.

        This is called after runtime parameters are updated via
        apply_runtime_overrides().
        """
        # Update temperature for probabilistic nodes
        if "temperature" in self.runtime and self.config.node_type == "probabilistic":
            for conv_layer in self.conv_layers:
                if hasattr(conv_layer, "first_layer_chunks"):
                    for chunk_layer in conv_layer.first_layer_chunks:
                        if hasattr(chunk_layer, "nodes"):
                            for node in chunk_layer.nodes:
                                if hasattr(node, "temperature"):
                                    node.temperature = self.runtime["temperature"]

                if hasattr(conv_layer, "trees"):
                    for tree in conv_layer.trees:
                        for layer in tree:
                            if hasattr(layer, "nodes"):
                                for node in layer.nodes:
                                    if hasattr(node, "temperature"):
                                        node.temperature = self.runtime["temperature"]

        # Update eval_mode for probabilistic nodes
        if "eval_mode" in self.runtime and self.config.node_type == "probabilistic":
            for conv_layer in self.conv_layers:
                if hasattr(conv_layer, "first_layer_chunks"):
                    for chunk_layer in conv_layer.first_layer_chunks:
                        if hasattr(chunk_layer, "nodes"):
                            for node in chunk_layer.nodes:
                                if hasattr(node, "eval_mode"):
                                    node.eval_mode = self.runtime["eval_mode"]

                if hasattr(conv_layer, "trees"):
                    for tree in conv_layer.trees:
                        for layer in tree:
                            if hasattr(layer, "nodes"):
                                for node in layer.nodes:
                                    if hasattr(node, "eval_mode"):
                                        node.eval_mode = self.runtime["eval_mode"]

        # Update output layer tau
        if "output_tau" in self.runtime:
            self.output_layer.tau = self.runtime["output_tau"]

        # Update layer-level parameters
        if "flip_probability" in self.runtime:
            for conv_layer in self.conv_layers:
                if hasattr(conv_layer, "flip_probability"):
                    conv_layer.flip_probability = self.runtime["flip_probability"]

        if "grad_stabilization" in self.runtime:
            for conv_layer in self.conv_layers:
                if hasattr(conv_layer, "grad_stabilization"):
                    conv_layer.grad_stabilization = self.runtime["grad_stabilization"]


# Register the model
REGISTRY.register_model("convolutional")(SimpleConvolutional)
