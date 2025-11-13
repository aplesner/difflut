"""
Global registry for DiffLUT components.
Provides a centralized system for registering and retrieving nodes, layers, and encoders.
"""

import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Type


class Registry:
    """
    Global registry for DiffLUT components.
    Manages registration and retrieval of nodes, layers, encoders, initializers, regularizers, and models.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Type] = {}
        self._layers: Dict[str, Type] = {}
        self._convolutional_layers: Dict[str, Type] = {}
        self._encoders: Dict[str, Type] = {}
        self._initializers: Dict[str, Callable] = {}
        self._regularizers: Dict[str, Callable] = {}
        self._models: Dict[str, Type] = {}

    # ==================== Node Registration ====================

    def register_node(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a node class.

        Args:
            name: Name to register the node under. If None, uses class name.

        Example:
            @registry.register_node("dwn")
            class DWNNode(BaseNode):
                pass
        """

        def decorator(cls: Type) -> Type:
            node_name = name if name is not None else cls.__name__
            if node_name in self._nodes:
                warnings.warn(
                    f"Node '{node_name}' is already registered and will be overwritten. "
                    f"This may lead to unexpected behavior if other code depends on the original implementation. "
                    f"Consider using a unique name or checking existing registrations with registry.list_nodes().",
                    UserWarning,
                    stacklevel=2,
                )
            self._nodes[node_name] = cls
            return cls

        return decorator

    def get_node(self, name: str) -> Type:
        """
        Get a registered node class by name.

        Args:
            name: Name of the node

        Returns:
            Node class

        Raises:
            ValueError: If node not found
        """
        if name not in self._nodes:
            raise ValueError(
                f"Node '{name}' not found. " f"Available nodes: {list(self._nodes.keys())}"
            )
        return self._nodes[name]

    def list_nodes(self) -> List[str]:
        """List all registered node names."""
        return list(self._nodes.keys())

    # ==================== Layer Registration ====================

    def register_layer(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a layer class.

        Args:
            name: Name to register the layer under. If None, uses class name.

        Example:
            @registry.register_layer("random")
            class RandomLayer(BaseLayer):
                pass
        """

        def decorator(cls: Type) -> Type:
            layer_name = name if name is not None else cls.__name__
            if layer_name in self._layers:
                warnings.warn(
                    f"Layer '{layer_name}' is already registered and will be overwritten. "
                    f"This may lead to unexpected behavior if other code depends on the original implementation. "
                    f"Consider using a unique name or checking existing registrations with registry.list_layers().",
                    UserWarning,
                    stacklevel=2,
                )
            self._layers[layer_name] = cls
            return cls

        return decorator

    def get_layer(self, name: str) -> Type:
        """
        Get a registered layer class by name.

        Args:
            name: Name of the layer

        Returns:
            Layer class

        Raises:
            ValueError: If layer not found
        """
        if name not in self._layers:
            raise ValueError(
                f"Layer '{name}' not found. " f"Available layers: {list(self._layers.keys())}"
            )
        return self._layers[name]

    def list_layers(self) -> List[str]:
        """List all registered layer names."""
        return list(self._layers.keys())

    # ==================== Convolutional Layer Registration ====================
    def register_convolutional_layer(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a convolutional layer class.

        Args:
            name: Name to register the convolutional layer under. If None, uses class name.

        Example:
            @registry.register_convolutional_layer("convolutional")
            class ConvolutionalLayer(BaseConvolutionalLayer):
                pass
        """

        def decorator(cls: Type) -> Type:
            conv_layer_name = name if name is not None else cls.__name__
            if conv_layer_name in self._convolutional_layers:
                warnings.warn(
                    f"Convolutional Layer '{conv_layer_name}' is already registered and will be overwritten. "
                    f"This may lead to unexpected behavior if other code depends on the original implementation. "
                    f"Consider using a unique name or checking existing registrations with registry.list_convolutional_layers().",
                    UserWarning,
                    stacklevel=2,
                )
            self._convolutional_layers[conv_layer_name] = cls
            return cls

        return decorator

    def get_convolutional_layer(self, name: str) -> Type:
        """
        Get a registered convolutional layer class by name.

        Args:
            name: Name of the convolutional layer

        Returns:
            Convolutional Layer class

        Raises:
            ValueError: If convolutional layer not found
        """
        if name not in self._convolutional_layers:
            raise ValueError(
                f"Convolutional Layer '{name}' not found. "
                f"Available layers: {list(self._convolutional_layers.keys())}"
            )
        return self._convolutional_layers[name]

    def list_convolutional_layers(self) -> List[str]:
        """List all registered convolutional layer names."""
        return list(self._convolutional_layers.keys())

    # ==================== Encoder Registration ====================

    def register_encoder(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to register an encoder class.

        Args:
            name: Name to register the encoder under. If None, uses class name.

        Example:
            @registry.register_encoder("thermometer")
            class ThermometerEncoder(BaseEncoder):
                pass
        """

        def decorator(cls: Type) -> Type:
            encoder_name = name if name is not None else cls.__name__
            if encoder_name in self._encoders:
                warnings.warn(
                    f"Encoder '{encoder_name}' is already registered and will be overwritten. "
                    f"This may lead to unexpected behavior if other code depends on the original implementation. "
                    f"Consider using a unique name or checking existing registrations with registry.list_encoders().",
                    UserWarning,
                    stacklevel=2,
                )
            self._encoders[encoder_name] = cls
            return cls

        return decorator

    def get_encoder(self, name: str) -> Type:
        """
        Get a registered encoder class by name.

        Args:
            name: Name of the encoder

        Returns:
            Encoder class

        Raises:
            ValueError: If encoder not found
        """
        if name not in self._encoders:
            raise ValueError(
                f"Encoder '{name}' not found. " f"Available encoders: {list(self._encoders.keys())}"
            )
        return self._encoders[name]

    def list_encoders(self) -> List[str]:
        """List all registered encoder names."""
        return list(self._encoders.keys())

    # ==================== Initializer Registration ====================

    def register_initializer(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to register an initializer function.

        Args:
            name: Name to register the initializer under. If None, uses function name.

        Example:
            @registry.register_initializer("xavier")
            def xavier_uniform_init(node, gain=1.0, **kwargs):
                # initialization logic
                pass
        """

        def decorator(func: Callable) -> Callable:
            init_name = (name if name is not None else func.__name__).lower()
            # Remove '_init' suffix if present for cleaner naming
            if init_name.endswith("_init"):
                init_name_clean = init_name[:-5]
                # Register both with and without suffix
                if init_name in self._initializers:
                    warnings.warn(
                        f"Initializer '{init_name}' is already registered and will be overwritten.",
                        UserWarning,
                        stacklevel=2,
                    )
                if init_name_clean in self._initializers:
                    warnings.warn(
                        f"Initializer '{init_name_clean}' is already registered and will be overwritten.",
                        UserWarning,
                        stacklevel=2,
                    )
                self._initializers[init_name] = func
                self._initializers[init_name_clean] = func
            else:
                if init_name in self._initializers:
                    warnings.warn(
                        f"Initializer '{init_name}' is already registered and will be overwritten.",
                        UserWarning,
                        stacklevel=2,
                    )
                self._initializers[init_name] = func
            return func

        return decorator

    def get_initializer(self, name: str) -> Callable:
        """
        Get a registered initializer function by name.

        Args:
            name: Name of the initializer (case-insensitive)

        Returns:
            Initializer function

        Raises:
            ValueError: If initializer not found
        """
        name_lower = name.lower()
        if name_lower not in self._initializers:
            raise ValueError(
                f"Initializer '{name}' not found. "
                f"Available initializers: {list(self._initializers.keys())}"
            )
        return self._initializers[name_lower]

    def list_initializers(self) -> List[str]:
        """List all registered initializer names."""
        return sorted(list(self._initializers.keys()))

    # ==================== Regularizer Registration ====================

    def register_regularizer(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a regularizer function.

        Args:
            name: Name to register the regularizer under. If None, uses function name.

        Example:
            @registry.register_regularizer("l1")
            def l1_regularizer(node, inputs=None):
                # regularization logic
                pass
        """

        def decorator(func: Callable) -> Callable:
            reg_name = (name if name is not None else func.__name__).lower()
            # Remove '_regularizer' suffix if present for cleaner naming
            if reg_name.endswith("_regularizer"):
                reg_name_clean = reg_name[: -len("_regularizer")]
                # Register both with and without suffix
                self._regularizers[reg_name_clean] = func
                self._regularizers[reg_name] = func
            else:
                self._regularizers[reg_name] = func
            return func

        return decorator

    def get_regularizer(self, name: str) -> Callable:
        """
        Get a registered regularizer function by name.

        Args:
            name: Name of the regularizer (case-insensitive)

        Returns:
            Regularizer function

        Raises:
            ValueError: If regularizer not found
        """
        name_lower = name.lower()
        if name_lower not in self._regularizers:
            raise ValueError(
                f"Regularizer '{name}' not found. "
                f"Available regularizers: {list(self._regularizers.keys())}"
            )
        return self._regularizers[name_lower]

    def list_regularizers(self) -> List[str]:
        """List all registered regularizer names."""
        return sorted(list(self._regularizers.keys()))

    # ==================== Builder Methods ====================

    def build_node(self, name: str, **kwargs) -> Any:
        """
        Build a node instance from its registered name.

        Args:
            name: Name of the node
            **kwargs: Arguments to pass to the node constructor

        Returns:
            Instance of the node
        """
        node_cls = self.get_node(name)
        return node_cls(**kwargs)

    def build_layer(self, name: str, **kwargs) -> Any:
        """
        Build a layer instance from its registered name.

        Args:
            name: Name of the layer
            **kwargs: Arguments to pass to the layer constructor

        Returns:
            Instance of the layer
        """
        layer_cls = self.get_layer(name)
        return layer_cls(**kwargs)

    def build_convolutional_layer(self, name: str, **kwargs) -> Any:
        """
        Build a convolutional layer instance from its registered name.

        Args:
            name: Name of the convolutional layer
            **kwargs: Arguments to pass to the convolutional layer constructor

        Returns:
            Instance of the convolutional layer
        """
        conv_layer_cls = self.get_convolutional_layer(name)
        return conv_layer_cls(**kwargs)

    def build_encoder(self, name: str, **kwargs) -> Any:
        """
        Build an encoder instance from its registered name.

        Args:
            name: Name of the encoder
            **kwargs: Arguments to pass to the encoder constructor

        Returns:
            Instance of the encoder
        """
        encoder_cls = self.get_encoder(name)
        return encoder_cls(**kwargs)

    # ==================== Model Registration ====================

    def register_model(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a model class.

        Args:
            name: Name to register the model under. If None, uses class name.

        Example:
            @registry.register_model("mnist_fc_8k")
            class MNISTSmall(BaseModel):
                pass
        """

        def decorator(cls: Type) -> Type:
            model_name = name if name is not None else cls.__name__
            if model_name in self._models:
                warnings.warn(
                    f"Model '{model_name}' is already registered and will be overwritten. "
                    f"This may lead to unexpected behavior if other code depends on the original implementation. "
                    f"Consider using a unique name or checking existing registrations with registry.list_models().",
                    UserWarning,
                    stacklevel=2,
                )
            self._models[model_name] = cls
            return cls

        return decorator

    def get_model(self, name: str) -> Type:
        """
        Get a registered model class by name.

        Args:
            name: Name of the model

        Returns:
            Model class

        Raises:
            ValueError: If model not found
        """
        if name not in self._models:
            raise ValueError(
                f"Model '{name}' not found. " f"Available models: {list(self._models.keys())}"
            )
        return self._models[name]

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def build_model(self, name: str, **kwargs) -> Any:
        """
        Build a model instance from its registered name.

        Args:
            name: Name of the model
            **kwargs: Arguments to pass to the model constructor

        Returns:
            Instance of the model
        """
        model_cls = self.get_model(name)
        return model_cls(**kwargs)
    
    def get_model_config(self, name: str) -> Optional[Any]:
        """
        Get the default configuration for a registered model.
        
        This looks for a default config associated with the model class,
        either as a class attribute or in the pretrained directory.
        
        Args:
            name: Name of the model
        
        Returns:
            Model configuration (if available), None otherwise
        """
        model_cls = self.get_model(name)
        
        # Check if model class has a default_config class attribute
        if hasattr(model_cls, 'default_config'):
            return model_cls.default_config
        
        return None

    # ==================== Utility Methods ====================

    def list_all(self) -> Dict[str, List[str]]:
        """List all registered components."""
        return {
            "nodes": self.list_nodes(),
            "layers": self.list_layers(),
            "convolutional_layers": self.list_convolutional_layers(),
            "encoders": self.list_encoders(),
            "initializers": self.list_initializers(),
            "regularizers": self.list_regularizers(),
            "models": self.list_models(),
        }
    
    def get_pretrained_models(self) -> Dict[str, List[str]]:
        """
        Get list of available pretrained models.
        
        This is a convenience method that delegates to the factory module
        to discover pretrained models in the pretrained directory.
        
        Returns:
            Dictionary mapping model type to list of pretrained model names
        """
        try:
            from .models.factory import list_pretrained_models
            return list_pretrained_models()
        except ImportError:
            warnings.warn(
                "Could not import factory module to list pretrained models. "
                "Make sure the models subpackage is properly installed.",
                UserWarning
            )
            return {}

    def __repr__(self) -> str:
        return (
            f"Registry(\n"
            f"  nodes={len(self._nodes)},\n"
            f"  layers={len(self._layers)},\n"
            f"  convolutional_layers={len(self._convolutional_layers)},\n"
            f"  encoders={len(self._encoders)},\n"
            f"  initializers={len(self._initializers)},\n"
            f"  regularizers={len(self._regularizers)},\n"
            f"  models={len(self._models)}\n"
            f")"
        )


# Global registry instance
REGISTRY = Registry()

# Convenience decorator aliases
register_node = REGISTRY.register_node
register_layer = REGISTRY.register_layer
register_convolutional_layer = REGISTRY.register_convolutional_layer
register_encoder = REGISTRY.register_encoder
register_initializer = REGISTRY.register_initializer
register_regularizer = REGISTRY.register_regularizer
register_model = REGISTRY.register_model
