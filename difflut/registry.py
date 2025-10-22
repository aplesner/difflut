"""
Global registry for DiffLUT components.
Provides a centralized system for registering and retrieving nodes, layers, and encoders.
"""

from typing import Dict, Type, Any, Optional, Callable
import inspect
import warnings


class Registry:
    """
    Global registry for DiffLUT components.
    Manages registration and retrieval of nodes, layers, encoders, initializers, and regularizers.
    """
    
    def __init__(self):
        self._nodes: Dict[str, Type] = {}
        self._layers: Dict[str, Type] = {}
        self._encoders: Dict[str, Type] = {}
        self._initializers: Dict[str, Callable] = {}
        self._regularizers: Dict[str, Callable] = {}
    
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
                    stacklevel=2
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
                f"Node '{name}' not found. "
                f"Available nodes: {list(self._nodes.keys())}"
            )
        return self._nodes[name]
    
    def list_nodes(self) -> list:
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
                    stacklevel=2
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
                f"Layer '{name}' not found. "
                f"Available layers: {list(self._layers.keys())}"
            )
        return self._layers[name]
    
    def list_layers(self) -> list:
        """List all registered layer names."""
        return list(self._layers.keys())
    
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
                    stacklevel=2
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
                f"Encoder '{name}' not found. "
                f"Available encoders: {list(self._encoders.keys())}"
            )
        return self._encoders[name]
    
    def list_encoders(self) -> list:
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
            init_name = name if name is not None else func.__name__
            # Remove '_init' suffix if present for cleaner naming
            if init_name.endswith('_init'):
                init_name_clean = init_name[:-5]
                # Register both with and without suffix
                if init_name in self._initializers:
                    warnings.warn(
                        f"Initializer '{init_name}' is already registered and will be overwritten.",
                        UserWarning,
                        stacklevel=2
                    )
                if init_name_clean in self._initializers:
                    warnings.warn(
                        f"Initializer '{init_name_clean}' is already registered and will be overwritten.",
                        UserWarning,
                        stacklevel=2
                    )
                self._initializers[init_name] = func
                self._initializers[init_name_clean] = func
            else:
                if init_name in self._initializers:
                    warnings.warn(
                        f"Initializer '{init_name}' is already registered and will be overwritten.",
                        UserWarning,
                        stacklevel=2
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
    
    def list_initializers(self) -> list:
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
            def l1_regularizer(node, num_samples=100):
                # regularization logic
                pass
        """
        def decorator(func: Callable) -> Callable:
            reg_name = name if name is not None else func.__name__
            # Remove '_regularizer' suffix if present for cleaner naming
            if reg_name.endswith('_regularizer'):
                reg_name_clean = reg_name[:-12]
                # Register both with and without suffix
                if reg_name in self._regularizers:
                    warnings.warn(
                        f"Regularizer '{reg_name}' is already registered and will be overwritten.",
                        UserWarning,
                        stacklevel=2
                    )
                if reg_name_clean in self._regularizers:
                    warnings.warn(
                        f"Regularizer '{reg_name_clean}' is already registered and will be overwritten.",
                        UserWarning,
                        stacklevel=2
                    )
                self._regularizers[reg_name] = func
                self._regularizers[reg_name_clean] = func
            else:
                if reg_name in self._regularizers:
                    warnings.warn(
                        f"Regularizer '{reg_name}' is already registered and will be overwritten.",
                        UserWarning,
                        stacklevel=2
                    )
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
    
    def list_regularizers(self) -> list:
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
    
    # ==================== Utility Methods ====================
    
    def list_all(self) -> Dict[str, list]:
        """List all registered components."""
        return {
            'nodes': self.list_nodes(),
            'layers': self.list_layers(),
            'encoders': self.list_encoders(),
            'initializers': self.list_initializers(),
            'regularizers': self.list_regularizers(),
        }
    
    def __repr__(self) -> str:
        return (
            f"Registry(\n"
            f"  nodes={len(self._nodes)},\n"
            f"  layers={len(self._layers)},\n"
            f"  encoders={len(self._encoders)},\n"
            f"  initializers={len(self._initializers)},\n"
            f"  regularizers={len(self._regularizers)}\n"
            f")"
        )


# Global registry instance
REGISTRY = Registry()

# Convenience decorator aliases
register_node = REGISTRY.register_node
register_layer = REGISTRY.register_layer
register_encoder = REGISTRY.register_encoder
register_initializer = REGISTRY.register_initializer
register_regularizer = REGISTRY.register_regularizer
