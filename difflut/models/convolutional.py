"""
SimpleConvolutional model for DiffLUT.

A config-based convolutional network implementation using LUT-based blocks.
Designed for processing image data with spatial structure preservation.

Similar to SimpleFeedForward but uses ConvolutionalBlock instead of dense layers.
"""

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

from ..blocks import BlockConfig, ConvolutionalLayer
from ..encoder import EncoderConfig
from ..heads import GroupSum
from ..layers.layer_config import LayerConfig
from ..nodes.node_config import NodeConfig
from ..registry import REGISTRY
from ..utils.warnings import warn_default_value
from .base_model import BaseLUTModel
from .model_config import ModelConfig

# ==================== Default Values ====================
# Module-level constants for all default parameters
# Used for initialization and warning messages

DEFAULT_ENCODER_CONFIG: Dict[str, Any] = {}
"""Default encoder configuration (empty dict, model must provide or error)."""

DEFAULT_INPUT_SIZE: Optional[int] = None
"""Default input size (None means infer from data during fit_encoder)."""

DEFAULT_NUM_CLASSES: Optional[int] = None
"""Default number of output classes (None means must be provided in config)."""

DEFAULT_LAYER_TYPE: str = "random"
"""Default layer type for building layers."""

DEFAULT_NODE_TYPE: str = "probabilistic"
"""Default node type within layers."""

DEFAULT_NODE_INPUT_DIM: int = 6
"""Default number of inputs to each node."""

DEFAULT_CONV_LAYER_WIDTHS: list = [32, 64, 128]
"""Default convolutional layer widths for convolutional architecture."""

DEFAULT_CONV_KERNEL_SIZE: int = 5
"""Default kernel size for convolutional layers."""

DEFAULT_CONV_STRIDE: int = 1
"""Default stride for convolutional layers."""

DEFAULT_CONV_PADDING: int = 0
"""Default padding for convolutional layers."""

DEFAULT_TREE_DEPTH: int = 2
"""Default tree depth for LUT blocks."""

DEFAULT_OUTPUT_TAU: float = 1.0
"""Default temperature parameter for output layer."""


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

        Parameters:
        - config: ModelConfig instance with all model parameters stored in params dict

        Expects config.params to contain:
        - encoder_config: Dict with encoder name and parameters
        - input_size: Optional input size (inferred from data if not provided)
        - num_classes: Number of output classes
        - layer_type: Type of layer implementation (default: 'random')
        - node_type: Type of node implementation (default: 'probabilistic')
        - node_input_dim: Number of inputs per node (default: 6)
        - conv_layer_widths: List of output channels for each convolutional layer
        - conv_kernel_size: Kernel size for convolutions (default: 5)
        - conv_stride: Stride for convolutions (default: 1)
        - conv_padding: Padding for convolutions (default: 0)
        - tree_depth: Depth of LUT tree (default: 2)
        """
        super().__init__(config)

        # Setup encoder - get from params dict
        encoder_config_data = config.params.get(
            "encoder_config", DEFAULT_ENCODER_CONFIG
        )
        if encoder_config_data == DEFAULT_ENCODER_CONFIG:
            warn_default_value("encoder_config", encoder_config_data, stacklevel=2)

        # Convert to EncoderConfig if needed (support both dict and EncoderConfig)
        if isinstance(encoder_config_data, dict):
            encoder_config = EncoderConfig.from_dict(encoder_config_data)
        else:
            encoder_config = encoder_config_data

        encoder_name = encoder_config.name
        encoder_params = {
            "num_bits": encoder_config.num_bits,
            "flatten": encoder_config.flatten,
        }
        encoder_params.update(encoder_config.extra_params)

        encoder_class = REGISTRY.get_encoder(encoder_name)
        self.encoder = encoder_class(**encoder_params)
        self.encoder_fitted = False

        # Input/output specifications - get from params dict
        self.input_size = config.params.get("input_size", DEFAULT_INPUT_SIZE)
        if self.input_size == DEFAULT_INPUT_SIZE:
            warn_default_value("input_size", self.input_size, stacklevel=2)

        self.num_classes = config.params.get("num_classes", DEFAULT_NUM_CLASSES)
        if self.num_classes == DEFAULT_NUM_CLASSES:
            warn_default_value("num_classes", self.num_classes, stacklevel=2)

        # Extract input channels
        # Handle both flattened size (int, backward compat) and shape tuple (C, H, W)
        if isinstance(self.input_size, (tuple, list)):
            self.in_channels = self.input_size[0]
        else:
            # For backward compatibility, assume 1 channel if given flat size
            # Actual channels will be inferred from data during fit_encoder
            self.in_channels = 1

        # Convolutional layers will be built after encoder is fitted
        self.conv_layers = nn.ModuleList()
        self.encoded_input_size = None

        # Convolutional parameters from params dict
        self.conv_kernel_size = config.params.get(
            "conv_kernel_size", DEFAULT_CONV_KERNEL_SIZE
        )
        if (
            self.conv_kernel_size == DEFAULT_CONV_KERNEL_SIZE
            and "conv_kernel_size" not in config.params
        ):
            warn_default_value("conv_kernel_size", self.conv_kernel_size, stacklevel=2)

        self.conv_stride = config.params.get("conv_stride", DEFAULT_CONV_STRIDE)
        if (
            self.conv_stride == DEFAULT_CONV_STRIDE
            and "conv_stride" not in config.params
        ):
            warn_default_value("conv_stride", self.conv_stride, stacklevel=2)

        self.conv_padding = config.params.get("conv_padding", DEFAULT_CONV_PADDING)
        if (
            self.conv_padding == DEFAULT_CONV_PADDING
            and "conv_padding" not in config.params
        ):
            warn_default_value("conv_padding", self.conv_padding, stacklevel=2)

        # Output layer (GroupSum)
        output_tau = config.runtime.get("output_tau", DEFAULT_OUTPUT_TAU)
        if output_tau == DEFAULT_OUTPUT_TAU and "output_tau" not in config.runtime:
            warn_default_value("output_tau", output_tau, stacklevel=2)
        self.output_layer = GroupSum(
            k=self.num_classes, tau=output_tau, use_randperm=False
        )

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
            # Only warn if config had explicit channel mismatch (not from flattened size)
            if (
                isinstance(self.input_size, (tuple, list))
                and self.in_channels != actual_channels
            ):
                print(
                    f"Warning: Input channels mismatch. "
                    f"Config: {self.in_channels}, Data: {actual_channels}"
                )
            # Update to actual channels from data (this is the real input dimensionality)
            self.in_channels = actual_channels

        # Move encoder to same device as data
        self.encoder = self.encoder.to(data.device)

        # Fit encoder
        print(
            f"Fitting encoder on {len(data_flat)} samples with shape {data_flat.shape}..."
        )
        self.encoder.fit(data_flat)

        # Get encoded size by encoding a sample
        sample_encoded = self.encoder.encode(data_flat[:1])
        self.encoded_input_size = sample_encoded.shape[1]
        print(f"Encoded input size: {self.encoded_input_size}")

        # Now build convolutional layers
        # Pass the data device to _build_conv_layers to ensure layers are built on the right device
        self._build_conv_layers(device=data.device)

        self.encoder_fitted = True

    def _build_conv_layers(self, device: Optional[torch.device] = None):
        """
        Build convolutional layers after encoder is fitted.

        Uses the registry to get layer and node classes, properly handles
        initialization and regularization.

        Args:
            device: Device to build layers on. If None, tries to infer from model.
        """
        config = self.config
        runtime = self.runtime

        # Determine the device for building layers
        if device is None:
            # Try encoder parameters first, then any model parameter, otherwise CPU
            try:
                device = next(self.encoder.parameters()).device
            except StopIteration:
                # Encoder has no parameters, try to get device from any model parameter
                try:
                    device = next(self.parameters()).device
                except StopIteration:
                    # No parameters at all, default to CPU
                    device = torch.device("cpu")

        # Set random seed to ensure reproducibility
        if config.seed is not None:
            torch.manual_seed(config.seed)

        # Get layer and node classes from registry - get from params dict with warnings
        layer_type = config.params.get("layer_type", DEFAULT_LAYER_TYPE)
        if layer_type == DEFAULT_LAYER_TYPE and "layer_type" not in config.params:
            warn_default_value("layer_type", layer_type, stacklevel=2)

        node_type = config.params.get("node_type", DEFAULT_NODE_TYPE)
        if node_type == DEFAULT_NODE_TYPE and "node_type" not in config.params:
            warn_default_value("node_type", node_type, stacklevel=2)

        node_input_dim = config.params.get("node_input_dim", DEFAULT_NODE_INPUT_DIM)
        if (
            node_input_dim == DEFAULT_NODE_INPUT_DIM
            and "node_input_dim" not in config.params
        ):
            warn_default_value("node_input_dim", node_input_dim, stacklevel=2)

        layer_class = REGISTRY.get_layer(layer_type)
        node_class = REGISTRY.get_node(node_type)

        # Create LayerConfig for layer-level parameters
        layer_cfg = LayerConfig(
            flip_probability=runtime.get("flip_probability", 0.0),
            grad_stabilization=runtime.get("grad_stabilization", "none"),
            grad_target_std=runtime.get("grad_target_std", 1.0),
            grad_subtract_mean=runtime.get("grad_subtract_mean", False),
            grad_epsilon=runtime.get("grad_epsilon", 1e-8),
        )

        # Get layer widths (output channels for each conv layer)
        conv_layer_widths = config.params.get(
            "conv_layer_widths", DEFAULT_CONV_LAYER_WIDTHS
        )
        if (
            conv_layer_widths == DEFAULT_CONV_LAYER_WIDTHS
            and "conv_layer_widths" not in config.params
        ):
            warn_default_value("conv_layer_widths", conv_layer_widths, stacklevel=2)

        # Get tree depth for convolutional blocks
        tree_depth = config.params.get("tree_depth", DEFAULT_TREE_DEPTH)
        if tree_depth == DEFAULT_TREE_DEPTH and "tree_depth" not in config.params:
            warn_default_value("tree_depth", tree_depth, stacklevel=2)

        # Build each convolutional layer
        in_channels = self.in_channels
        for layer_idx, out_channels in enumerate(conv_layer_widths):
            # Build node configuration
            fan_in = node_input_dim

            # For conv layers, fan_out is harder to calculate precisely
            # Use a reasonable heuristic
            receptive_field = self.conv_kernel_size
            receptive_area = receptive_field * receptive_field
            fan_out = (out_channels * node_input_dim) / (in_channels * receptive_area)

            node_kwargs = self._build_node_config(fan_in=fan_in, fan_out=fan_out)

            # Create BlockConfig for convolutional block
            block_config = BlockConfig(
                block_type="convolutional",
                seed=config.seed + layer_idx,  # Different seed for each layer
                tree_depth=tree_depth,
                in_channels=in_channels,
                out_channels=out_channels,
                receptive_field=self.conv_kernel_size,
                stride=self.conv_stride,
                padding=self.conv_padding,
                patch_chunk_size=runtime.get(
                    "patch_chunk_size", 100
                ),  # Default: 100 patches per chunk
                n_inputs_per_node=node_input_dim,
                node_kwargs=node_kwargs.to_dict(),
                layer_config=layer_cfg.to_dict() if layer_cfg else {},
                flip_probability=layer_cfg.flip_probability if layer_cfg else 0.0,
                grad_stabilization=(
                    layer_cfg.grad_stabilization if layer_cfg else "none"
                ),
                grad_target_std=layer_cfg.grad_target_std if layer_cfg else 1.0,
                grad_subtract_mean=layer_cfg.grad_subtract_mean if layer_cfg else False,
                grad_epsilon=layer_cfg.grad_epsilon if layer_cfg else 1e-8,
                grouped_connections=runtime.get("grouped_connections", False),
                ensure_full_coverage=runtime.get("ensure_full_coverage", False),
            )

            # Create convolutional block
            conv_layer = ConvolutionalLayer(
                config=block_config,
                node_type=node_class,
                layer_type=layer_class,
            )

            # Move layer to same device as model
            conv_layer = conv_layer.to(device)

            self.conv_layers.append(conv_layer)
            in_channels = out_channels

        # Also move output_layer to same device
        self.output_layer = self.output_layer.to(device)

        print(
            f"Built {len(self.conv_layers)} convolutional layers with widths {conv_layer_widths}"
        )

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
        # NOTE: use_cuda parameter is removed - CUDA kernels are auto-selected based on device
        # Just do model.cuda() to use GPU kernels, model.cpu() to use CPU kernels
        node_type = config.params.get("node_type", "probabilistic")

        if node_type in ["dwn", "hybrid", "dwn_stable"]:
            pass  # No use_cuda - device determines kernel selection

        elif node_type == "probabilistic":
            extra_params["temperature"] = runtime.get("temperature", 1.0)
            extra_params["eval_mode"] = runtime.get("eval_mode", "expectation")
            # No use_cuda - device determines kernel selection

        elif node_type == "fourier":
            # No use_cuda - device determines kernel selection
            extra_params["use_all_frequencies"] = runtime.get(
                "use_all_frequencies", True
            )
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
            input_dim=config.params.get("node_input_dim", 6),
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

        # Ensure input is on the same device as model parameters
        device = next(self.parameters()).device if list(self.parameters()) else "cpu"
        x = x.to(device)

        # Ensure encoder is on the same device as input
        self.encoder = self.encoder.to(device)

        # Ensure input is 4D (batch, channels, height, width)
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (N, C, H, W), got shape {x.shape}. "
                "SimpleConvolutional expects image data."
            )

        batch_size = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]

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
        spatial_pixels = min_spatial + encoded_size // self.in_channels
        spatial_h = spatial_w = int(spatial_pixels**0.5) + 1

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
            x = x_encoded_padded.view(
                batch_size, self.in_channels, spatial_h, spatial_w
            )
        else:
            x = x_encoded.view(batch_size, self.in_channels, spatial_h, spatial_w)
        # Pass through convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Flatten output from convolutional layers to (N, features)
        x = x.reshape(batch_size, -1)

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
