"""
SimpleFeedForward model for DiffLUT.

A clean, config-based feedforward network implementation using the new
model infrastructure. Simpler than LayeredFeedForward with focus on
standardized configuration and pretrained model support.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..layers.layer_config import LayerConfig
from ..nodes.node_config import NodeConfig
from ..registry import REGISTRY
from ..utils import GroupSum
from .base_model import BaseLUTModel
from .model_config import ModelConfig


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

        Args:
            config: ModelConfig instance with all model parameters
        """
        super().__init__(config)

        # Setup encoder
        encoder_config = config.encoder_config
        encoder_name = encoder_config.get("name", "thermometer")
        encoder_params = {k: v for k, v in encoder_config.items() if k != "name"}

        encoder_class = REGISTRY.get_encoder(encoder_name)
        self.encoder = encoder_class(**encoder_params)
        self.encoder_fitted = False

        # Input/output sizes
        self.input_size = config.input_size
        self.num_classes = config.num_classes

        # Layers will be built after encoder is fitted
        self.layers = nn.ModuleList()
        self.encoded_input_size = None

        # Output layer (GroupSum)
        output_tau = config.runtime.get("output_tau", 1.0)
        self.output_layer = GroupSum(k=self.num_classes, tau=output_tau, use_randperm=False)

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
            self.config.input_size = self.input_size

        # Fit encoder
        print(f"Fitting encoder on {len(data_flat)} samples with shape {data_flat.shape}...")
        self.encoder.fit(data_flat)

        # Get encoded size by encoding a sample
        sample_encoded = self.encoder.encode(data_flat[:1])
        self.encoded_input_size = sample_encoded.shape[1]
        print(f"Encoded input size: {self.encoded_input_size}")

        # Now build layers with correct input size
        self._build_layers()

        # Move layers to the same device as the input data
        # This ensures GPU compatibility when model.cuda() is called before fit_encoder()
        if data.is_cuda:
            for layer in self.layers:
                layer.to(data.device)

        self.encoder_fitted = True

    def _build_layers(self):
        """
        Build layers after encoder is fitted.

        Uses the registry to get layer and node classes, properly handles
        initialization and regularization.
        """
        config = self.config
        current_size = self.encoded_input_size

        # Set random seed to ensure reproducibility
        # This is critical for CPU/GPU consistency since layers are built
        # after model initialization (inside fit_encoder)
        if config.seed is not None:
            torch.manual_seed(config.seed)

        # Get layer and node classes from registry
        layer_class = REGISTRY.get_layer(config.layer_type)
        node_class = REGISTRY.get_node(config.node_type)

        # Create LayerConfig for layer-level parameters
        layer_cfg = LayerConfig(
            flip_probability=config.runtime.get("flip_probability", 0.0),
            grad_stabilization=config.runtime.get("grad_stabilization", "none"),
            grad_target_std=config.runtime.get("grad_target_std", 1.0),
            grad_subtract_mean=config.runtime.get("grad_subtract_mean", False),
            grad_epsilon=config.runtime.get("grad_epsilon", 1e-8),
        )

        # Build each layer
        for i, layer_width in enumerate(config.layer_widths):
            # Calculate fan_in and fan_out for initialization
            fan_in = config.node_input_dim

            # fan_out: how many inputs does each node in the next layer connect to?
            if i + 1 < len(config.layer_widths):
                next_layer_width = config.layer_widths[i + 1]
                fan_out = (next_layer_width * config.node_input_dim) / layer_width
            else:
                # Last layer connects to output
                fan_out = (self.num_classes * config.node_input_dim) / layer_width

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

            self.layers.append(layer)
            current_size = layer_width

        print(f"Built {len(self.layers)} layers: {config.layer_widths}")

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
        node_type = config.node_type

        if node_type in ["dwn", "hybrid", "dwn_stable"]:
            pass  # No use_cuda - device determines kernel selection

        elif node_type == "probabilistic":
            extra_params["temperature"] = runtime.get("temperature", 1.0)
            extra_params["eval_mode"] = runtime.get("eval_mode", "expectation")
            # No use_cuda - device determines kernel selection

        elif node_type == "fourier":
            # No use_cuda - device determines kernel selection
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
            raise RuntimeError("Encoder must be fitted before encoding. Call fit_encoder() first.")
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

        # Flatten input if needed
        batch_size = x.shape[0]
        if x.dim() > 2:
            x = x.view(batch_size, -1)

        # Move encoder to same device as input
        self.encoder.to(x.device)

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
                    layer.layer_config.flip_probability = self.runtime["flip_probability"]

        if "grad_stabilization" in self.runtime:
            for layer in self.layers:
                if hasattr(layer, "layer_config"):
                    layer.layer_config.grad_stabilization = self.runtime["grad_stabilization"]


# Register the model
REGISTRY.register_model("feedforward")(SimpleFeedForward)
