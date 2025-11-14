"""
Model configuration dataclass for DiffLUT models.

Provides a unified configuration system that separates structural parameters
(which must match pretrained weights) from runtime parameters (which can be
safely overridden).
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Union
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Configuration for DiffLUT models.
    
    Separates structural parameters (must match pretrained weights) from
    runtime parameters (safe to override at runtime).
    
    Structural Parameters (define model architecture):
        - model_type: Type of model (feedforward, convnet, etc.)
        - layer_type: Type of layers (random, residual, etc.)
        - node_type: Type of nodes (dwn, probabilistic, etc.)
        - encoder_config: Encoder configuration (type, bits, etc.)
        - node_input_dim: Input dimension for nodes
        - layer_widths: Width of each hidden layer
        - num_classes: Number of output classes
        - dataset: Target dataset (for reference)
    
    Runtime Parameters (safe to override):
        - Stored in the 'runtime' dict
        - Examples: dropout, bitflip_prob, temperature, eval_mode
    
    Pretrained Information:
        - pretrained: Whether this is a pretrained model
        - pretrained_name: Name of the pretrained model
    """
    
    # ==================== Structural Parameters ====================
    # These define the model architecture and must match pretrained weights
    
    model_type: str
    layer_type: str
    node_type: str
    encoder_config: Dict[str, Any]
    node_input_dim: int
    layer_widths: List[int]
    num_classes: int
    dataset: Optional[str] = None
    
    # Additional structural parameters (optional)
    input_size: Optional[int] = None  # Raw input size (before encoding)
    seed: int = 42
    
    # ==================== Runtime Parameters ====================
    # These can be safely overridden without affecting model structure
    
    runtime: Dict[str, Any] = field(default_factory=dict)
    
    # Example runtime parameters that might be in runtime dict:
    # - dropout: float (dropout probability)
    # - bitflip_prob: float (bit flip probability)
    # - temperature: float (temperature for probabilistic nodes)
    # - eval_mode: str (evaluation mode for probabilistic nodes)
    # - grad_stabilization: str (gradient stabilization method)
    # - grad_target_std: float (target std for gradient stabilization)
    
    # ==================== Pretrained Information ====================
    
    pretrained: bool = False
    pretrained_name: Optional[str] = None
    
    # ==================== Serialization ====================
    
    @staticmethod
    def from_yaml(path: str) -> "ModelConfig":
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            ModelConfig instance
            
        Example YAML:
            model_type: feedforward
            layer_type: random
            node_type: probabilistic
            encoder_config:
              name: thermometer
              num_bits: 4
              feature_wise: true
            node_input_dim: 6
            layer_widths: [1024, 1000]
            num_classes: 10
            dataset: mnist
            runtime:
              temperature: 1.0
              eval_mode: expectation
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        # Ensure required fields are present
        required_fields = [
            'model_type', 'layer_type', 'node_type', 'encoder_config',
            'node_input_dim', 'layer_widths', 'num_classes'
        ]
        for field_name in required_fields:
            if field_name not in data:
                raise ValueError(f"Missing required field in config: {field_name}")
        
        return ModelConfig(**data)
    
    def to_yaml(self, path: str):
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save YAML configuration file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, default_flow_style=False, sort_keys=False)
    
    def save_to_pretrained(self, name: str, pretrained_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Save configuration to the pretrained models directory.
        
        Saves config to: pretrained/<model_type>/<name>.yaml
        
        Args:
            name: Name to save the model as (e.g., "cifar10_ffn_baseline")
            pretrained_dir: Base directory for pretrained models.
                           If None, uses difflut/models/pretrained
        
        Returns:
            Path to the saved configuration file
            
        Example:
            >>> config = ModelConfig(...)
            >>> config_path = config.save_to_pretrained("cifar10_ffn_baseline")
            >>> # Saves to: pretrained/feedforward/cifar10_ffn_baseline.yaml
        """
        if pretrained_dir is None:
            # Use default pretrained directory in models package
            pretrained_dir = Path(__file__).parent / "pretrained"
        else:
            pretrained_dir = Path(pretrained_dir)
        
        # Create model-type-specific subdirectory
        model_type_dir = pretrained_dir / self.model_type
        model_type_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = model_type_dir / f"{name}.yaml"
        self.to_yaml(str(config_path))
        
        return config_path
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        return asdict(self)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ModelConfig":
        """
        Create configuration from dictionary.
        
        Args:
            data: Dictionary with configuration parameters
            
        Returns:
            ModelConfig instance
        """
        return ModelConfig(**data)
    
    # ==================== Utility Methods ====================
    
    def get_layer_depth(self) -> int:
        """Get the number of hidden layers."""
        return len(self.layer_widths)
    
    def get_total_params_estimate(self) -> int:
        """
        Estimate total number of parameters (rough calculation).
        
        Returns:
            Estimated parameter count
        """
        # This is a rough estimate for feedforward models
        # Actual count depends on node implementation
        total = 0
        
        # Encoder parameters (usually negligible or non-trainable)
        # Assume encoded_size ≈ input_size * encoder_bits (rough estimate)
        
        # Node parameters
        lut_size = 2 ** self.node_input_dim
        
        # First layer: encoded_input → layer_widths[0]
        if self.input_size:
            encoded_size = self.input_size * self.encoder_config.get('num_bits', 4)
            num_nodes_first = self.layer_widths[0]
            # Each node connects to node_input_dim inputs
            total += num_nodes_first * lut_size
        
        # Hidden layers
        for i in range(len(self.layer_widths) - 1):
            num_nodes = self.layer_widths[i + 1]
            total += num_nodes * lut_size
        
        return total
    
    def is_compatible_with_weights(self, other: "ModelConfig") -> bool:
        """
        Check if this config is compatible with weights from another config.
        
        Two configs are compatible if all structural parameters match.
        
        Args:
            other: Another ModelConfig instance
            
        Returns:
            True if configs are compatible (weights can be shared)
        """
        structural_fields = [
            'model_type', 'layer_type', 'node_type', 'encoder_config',
            'node_input_dim', 'layer_widths', 'num_classes', 'input_size'
        ]
        
        for field_name in structural_fields:
            if getattr(self, field_name) != getattr(other, field_name):
                return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation of config."""
        runtime_str = ", ".join(f"{k}={v}" for k, v in self.runtime.items())
        return (
            f"ModelConfig(\n"
            f"  model_type={self.model_type},\n"
            f"  node_type={self.node_type},\n"
            f"  layer_widths={self.layer_widths},\n"
            f"  num_classes={self.num_classes},\n"
            f"  runtime=[{runtime_str}]\n"
            f")"
        )
