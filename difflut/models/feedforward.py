"""
SimpleFeedForward model for DiffLUT.

A clean, config-based feedforward network implementation using the new
model infrastructure. Simpler than LayeredFeedForward with focus on
standardized configuration and pretrained model support.
"""

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

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

DEFAULT_LAYER_WIDTHS: list = [1024, 1000]
"""Default layer widths for feedforward architecture."""

DEFAULT_OUTPUT_TAU: float = 1.0
"""Default temperature parameter for output layer."""


class SimpleFeedForward(BaseLUTModel):
    """
    Simple feedforward network using DiffLUT nodes.

    This model:
    - Uses config-based initialization for reproducibility
    - Supports pretrained weights via ModelConfig
    - Handles runtime parameter overrides
    - Automatically fits encoder on training data
    - Manages node initialization and regularization via registry

    Differences from LayeredFeedForward:
    - Cleaner config interface (ModelConfig instead of nested dicts)
    - Better separation of structural vs runtime parameters
    - Simplified encoder handling
    - Standardized pretrained model support
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the SimpleFeedForward model.

        Parameters:
        - config: ModelConfig instance with all model parameters stored in params dict

        Expects config.params to contain:
        - encoder_config: Dict with encoder name and parameters
        - input_size: Optional input size (inferred from data if not provided)
        - num_classes: Number of output classes
        - layer_type: Type of layer implementation (default: 'random')
        - node_type: Type of node implementation (default: 'probabilistic')
        - node_input_dim: Number of inputs per node (default: 6)
        - layer_widths: List of output dimensions for each layer
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

        # Input/output sizes - get from params dict
        self.input_size = config.params.get("input_size", DEFAULT_INPUT_SIZE)
        if self.input_size == DEFAULT_INPUT_SIZE:
            warn_default_value("input_size", self.input_size, stacklevel=2)

        self.num_classes = config.params.get("num_classes", DEFAULT_NUM_CLASSES)
        if self.num_classes == DEFAULT_NUM_CLASSES:
            warn_default_value("num_classes", self.num_classes, stacklevel=2)

        # Layers will be built after encoder is fitted
        self.layers = nn.ModuleList()
        self.encoded_input_size = None

        # Output layer (GroupSum)
        output_tau = config.runtime.get("output_tau", DEFAULT_OUTPUT_TAU)
        if output_tau == DEFAULT_OUTPUT_TAU and "output_tau" not in config.runtime:
            warn_default_value("output_tau", output_tau, stacklevel=2)
        self.output_layer = GroupSum(
            k=self.num_classes, tau=output_tau, use_randperm=False
        )

    def fit_encoder(self, data: torch.Tensor):
        """
        Fit the encoder on training data and build layers.

        Args:
            data: Training data tensor (N, ...) - will be flattened if needed
        """
        if self.encoder_fitted:
            print("Warning: Encoder already fitted, skipping...")
            return

        # Flatten data if needed
        batch_size = data.shape[0]
        if data.dim() > 2:
            data_flat = data.view(batch_size, -1)
        else:
            data_flat = data

        # Update input_size if not set
        if self.input_size is None:
            self.input_size = data_flat.shape[1]
            self.config.params["input_size"] = self.input_size

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

        # Now build layers with correct input size
        # Pass the data device to _build_layers to ensure layers are built on the right device
        self._build_layers(device=data.device)

        self.encoder_fitted = True

    def _build_layers(self, device: Optional[torch.device] = None):
        """
        Build layers after encoder is fitted.

        Uses the registry to get layer and node classes, properly handles
        initialization and regularization.

        Args:
            device: Device to build layers on. If None, tries to infer from model.
        """
        config = self.config
        current_size = self.encoded_input_size

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
        # This is critical for CPU/GPU consistency since layers are built
        # after model initialization (inside fit_encoder)
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

        layer_widths = config.params.get("layer_widths", DEFAULT_LAYER_WIDTHS)
        if layer_widths == DEFAULT_LAYER_WIDTHS and "layer_widths" not in config.params:
            warn_default_value("layer_widths", layer_widths, stacklevel=2)

        layer_class = REGISTRY.get_layer(layer_type)
        node_class = REGISTRY.get_node(node_type)

        # Create LayerConfig for layer-level parameters
        layer_cfg = LayerConfig(
            flip_probability=config.runtime.get("flip_probability", 0.0),
            grad_stabilization=config.runtime.get("grad_stabilization", "none"),
            grad_target_std=config.runtime.get("grad_target_std", 1.0),
            grad_subtract_mean=config.runtime.get("grad_subtract_mean", False),
            grad_epsilon=config.runtime.get("grad_epsilon", 1e-8),
        )

        # Build each layer
        for i, layer_width in enumerate(layer_widths):
            # Calculate fan_in and fan_out for initialization
            fan_in = node_input_dim

            # fan_out: how many inputs does each node in the next layer connect to?
            if i + 1 < len(layer_widths):
                next_layer_width = layer_widths[i + 1]
                fan_out = (next_layer_width * node_input_dim) / layer_width
            else:
                # Last layer connects to output
                fan_out = (self.num_classes * node_input_dim) / layer_width

            # Build node configuration
            node_kwargs = self._build_node_config(fan_in=fan_in, fan_out=fan_out)

            # Create layer
            layer = layer_class(
                input_size=current_size,
                output_size=layer_width,
                node_type=node_class,
                node_kwargs=node_kwargs,
                seed=config.seed,
                layer_config=layer_cfg,
            )

            # Move layer to same device as model
            layer = layer.to(device)

            self.layers.append(layer)
            current_size = layer_width

        # Also move output_layer to same device
        self.output_layer = self.output_layer.to(device)

        print(f"Built {len(self.layers)} layers: {layer_widths}")

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
            output_dim=1,  # Always 1 for standard feedforward
            init_fn=init_fn,
            init_kwargs=init_kwargs if init_kwargs else None,
            regularizers=regularizers if regularizers else None,
            extra_params=extra_params,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input data using the fitted encoder.

        Args:
            x: Input tensor (N, ...) where ... represents feature dimensions

        Returns:
            Encoded tensor (N, encoded_size)
        """
        if not self.encoder_fitted:
            raise RuntimeError(
                "Encoder must be fitted before encoding. Call fit_encoder() first."
            )
        return self.encoder.encode(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the entire network.

        Args:
            x: Input tensor (N, ...) - raw input (images, sequences, etc.)
               Will be flattened, encoded, and passed through DiffLUT layers

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

        # Flatten input if needed
        batch_size = x.shape[0]
        if x.dim() > 2:
            x = x.view(batch_size, -1)

        # Encode input
        x = self.encoder.encode(x)

        # Clamp to valid range [0, 1]
        x = torch.clamp(x, 0, 1)

        # Pass through DiffLUT layers
        for layer in self.layers:
            x = layer(x)

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
            for layer in self.layers:
                if hasattr(layer, "nodes"):
                    for node in layer.nodes:
                        if hasattr(node, "temperature"):
                            node.temperature = self.runtime["temperature"]

        # Update eval_mode for probabilistic nodes
        if "eval_mode" in self.runtime and self.config.node_type == "probabilistic":
            for layer in self.layers:
                if hasattr(layer, "nodes"):
                    for node in layer.nodes:
                        if hasattr(node, "eval_mode"):
                            node.eval_mode = self.runtime["eval_mode"]

        # Update output layer tau
        if "output_tau" in self.runtime:
            self.output_layer.tau = self.runtime["output_tau"]

        # Update layer-level parameters
        if "flip_probability" in self.runtime:
            for layer in self.layers:
                if hasattr(layer, "layer_config"):
                    layer.layer_config.flip_probability = self.runtime[
                        "flip_probability"
                    ]

        if "grad_stabilization" in self.runtime:
            for layer in self.layers:
                if hasattr(layer, "layer_config"):
                    layer.layer_config.grad_stabilization = self.runtime[
                        "grad_stabilization"
                    ]


# Register the model
REGISTRY.register_model("feedforward")(SimpleFeedForward)
