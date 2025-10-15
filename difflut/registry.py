"""
Global registry for DiffLUT components.
Provides a centralized system for registering and retrieving nodes, layers, and encoders.
"""

from typing import Dict, Type, Any, Optional, Callable
import inspect


class Registry:
    """
    Global registry for DiffLUT components.
    Manages registration and retrieval of nodes, layers, and encoders.
    """
    
    def __init__(self):
        self._nodes: Dict[str, Type] = {}
        self._layers: Dict[str, Type] = {}
        self._encoders: Dict[str, Type] = {}
    
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
                raise ValueError(f"Node '{node_name}' is already registered")
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
                raise ValueError(f"Layer '{layer_name}' is already registered")
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
                raise ValueError(f"Encoder '{encoder_name}' is already registered")
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
        }
    
    def __repr__(self) -> str:
        return (
            f"Registry(\n"
            f"  nodes={len(self._nodes)},\n"
            f"  layers={len(self._layers)},\n"
            f"  encoders={len(self._encoders)}\n"
            f")"
        )


# Global registry instance
REGISTRY = Registry()

# Convenience decorator aliases
register_node = REGISTRY.register_node
register_layer = REGISTRY.register_layer
register_encoder = REGISTRY.register_encoder
