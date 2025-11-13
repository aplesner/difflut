"""
DiffLUT Model Zoo

Ready-to-use architectures for common tasks:
- MNIST: Fully connected models for fast prototyping and FPGA export
- CIFAR-10: Convolutional models showcasing color image processing
- Comparison: Models demonstrating training techniques (bit flipping, grad norm, etc.)

Example usage:

    from difflut.models import MNISTLinearSmall, CIFAR10Conv, get_model, list_models
    from difflut import REGISTRY
    
    # Get model by name from registry
    model = REGISTRY.build_model('mnist_fc_8k_linear')
    
    # Or use the convenience function
    model = get_model('mnist_fc_8k_linear')
    
    # List all available models
    print(list_models())
    
    # Create model directly
    model = MNISTLinearSmall()
    
    # Fit encoder and train
    model.fit_encoder(train_images)
    outputs = model(test_images)

Models are designed to be:
- Self-contained: All parameters hardcoded for reproducibility
- Educational: Show different DiffLUT capabilities
- Production-ready: Suitable for benchmarking and export
"""

from difflut import REGISTRY

# Import base class
from .base_model import BaseModel

# Import MNIST models
from .mnist import (
    MNISTSmallBase,
    MNISTLinearSmall,
    MNISTDWNSmall,
)

# Import CIFAR-10 models
from .cifar10 import CIFAR10Conv

# Import comparison/utility models
from .comparison import (
    ComparisonModelBase,
    # Bit flip variants
    MNISTBitFlipNone,
    MNISTBitFlip5,
    MNISTBitFlip10,
    MNISTBitFlip20,
    # Gradient norm variants
    MNISTGradNormNone,
    MNISTGradNormLayerwise,
    MNISTGradNormBatchwise,
    MNISTGradNormLayerwiseWithMean,
    # Residual init
    MNISTResidualInit,
)

# Register all models with the global registry
REGISTRY.register_model('mnist_fc_8k_linear')(MNISTLinearSmall)
REGISTRY.register_model('mnist_fc_8k_dwn')(MNISTDWNSmall)
REGISTRY.register_model('cifar10_conv')(CIFAR10Conv)
REGISTRY.register_model('mnist_bitflip_none')(MNISTBitFlipNone)
REGISTRY.register_model('mnist_bitflip_5')(MNISTBitFlip5)
REGISTRY.register_model('mnist_bitflip_10')(MNISTBitFlip10)
REGISTRY.register_model('mnist_bitflip_20')(MNISTBitFlip20)
REGISTRY.register_model('mnist_gradnorm_none')(MNISTGradNormNone)
REGISTRY.register_model('mnist_gradnorm_layerwise')(MNISTGradNormLayerwise)
REGISTRY.register_model('mnist_gradnorm_batchwise')(MNISTGradNormBatchwise)
REGISTRY.register_model('mnist_gradnorm_layerwise_mean')(MNISTGradNormLayerwiseWithMean)
REGISTRY.register_model('mnist_residual_init')(MNISTResidualInit)

__all__ = [
    # Base
    "BaseModel",
    
    # MNIST models
    "MNISTSmallBase",
    "MNISTLinearSmall",
    "MNISTDWNSmall",
    
    # CIFAR-10 models
    "CIFAR10Conv",
    
    # Comparison models
    "ComparisonModelBase",
    "MNISTBitFlipNone",
    "MNISTBitFlip5",
    "MNISTBitFlip10",
    "MNISTBitFlip20",
    "MNISTGradNormNone",
    "MNISTGradNormLayerwise",
    "MNISTGradNormBatchwise",
    "MNISTGradNormLayerwiseWithMean",
    "MNISTResidualInit",
    
    # Factory functions and registry
    "get_model",
    "list_models",
    "model_info",
    "print_models_table",
]


def get_model(
    name: str,
    pretrained: bool = False,
    **kwargs
) -> BaseModel:
    """
    Get a model from the model zoo using the registry.
    
    Args:
        name: Model identifier (see list_models() for options)
        pretrained: Whether to load pretrained weights (not yet implemented)
        **kwargs: Model-specific parameters (override defaults)
    
    Returns:
        Instantiated model
    
    Raises:
        ValueError: If model not found
    
    Example:
        >>> model = get_model('mnist_fc_8k_linear')
        >>> model = get_model('cifar10_conv', node_type='dwn_stable')
    """
    try:
        model_class = REGISTRY.get_model(name)
    except ValueError:
        raise ValueError(
            f"Model '{name}' not found. "
            f"Available models: {list_models()}"
        )
    
    # Handle pretrained weights (not yet implemented)
    if pretrained:
        raise NotImplementedError(
            "Pretrained weights not yet available. "
            "Models must be trained from scratch."
        )
    
    # Instantiate with kwargs if provided
    if kwargs:
        return model_class(**kwargs)
    else:
        return model_class()


def list_models() -> list:
    """
    List all available models in the zoo.
    
    Returns:
        List of model identifiers
    
    Example:
        >>> models = list_models()
        >>> print(f"Available: {models}")
    """
    return sorted(REGISTRY.list_models())


def model_info(name: str) -> dict:
    """
    Get information about a specific model.
    
    Args:
        name: Model identifier
    
    Returns:
        Dictionary with model metadata
    
    Raises:
        ValueError: If model not found
    """
    try:
        model_class = REGISTRY.get_model(name)
    except ValueError:
        raise ValueError(f"Model '{name}' not found")
    
    # Create a temporary instance to get info
    # Note: This doesn't require encoder fitting
    try:
        model = model_class()
        return {
            'name': name,
            'class': model_class.__name__,
            'input_size': model.input_size,
            'num_classes': model.num_classes,
            'model_name': model.model_name,
            'doc': model_class.__doc__
        }
    except Exception as e:
        return {
            'name': name,
            'class': model_class.__name__,
            'error': str(e)
        }


def print_models_table() -> None:
    """
    Print a formatted table of all available models.
    
    Useful for understanding the model zoo structure.
    """
    import textwrap
    
    print("\n" + "="*80)
    print("DiffLUT Model Zoo - Available Models")
    print("="*80)
    
    # Group models by category
    models_list = list_models()
    categories = {
        'MNIST (Fully Connected)': [m for m in models_list if 'mnist' in m and 'bitflip' not in m and 'gradnorm' not in m and 'residual' not in m],
        'CIFAR-10 (Convolutional)': [m for m in models_list if 'cifar' in m],
        'Bit Flipping Comparison': [m for m in models_list if 'bitflip' in m],
        'Gradient Normalization': [m for m in models_list if 'gradnorm' in m],
        'Residual Initialization': [m for m in models_list if 'residual' in m],
    }
    
    for category, models in categories.items():
        if models:
            print(f"\n{category}:")
            print("-" * 80)
            for model_name in models:
                info = model_info(model_name)
                print(f"  {model_name:<30} - {info['class']}")
                if 'doc' in info and info['doc']:
                    doc_lines = info['doc'].split('\n')[:2]
                    for line in doc_lines:
                        wrapped = textwrap.fill(
                            line.strip(),
                            width=76,
                            initial_indent='      ',
                            subsequent_indent='      '
                        )
                        if wrapped.strip():
                            print(wrapped)
    
    print("\n" + "="*80)
    print(f"Total: {len(list_models())} models available")
    print("="*80 + "\n")
