"""
Demonstration of the AdaptiveLayer and its parameter efficiency.

This script compares the AdaptiveLayer with the LearnableLayer to show
how sparse connections and weight sharing reduce the number of parameters.
"""

import torch
import sys
sys.path.insert(0, '..')

from difflut.layers import LearnableLayer, AdaptiveLayer
from difflut.nodes import LinearLUTNode


def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def compare_layers():
    """Compare LearnableLayer and AdaptiveLayer parameter counts."""
    
    # Configuration
    input_size = 784  # MNIST input size
    output_size = 128  # Number of nodes
    n = 6  # Inputs per LUT
    
    print("=" * 70)
    print("Adaptive Layer Parameter Efficiency Demonstration")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input size: {input_size}")
    print(f"  Output size (number of nodes): {output_size}")
    print(f"  Inputs per LUT (n): {n}")
    print()
    
    # Create LearnableLayer (baseline)
    learnable_layer = LearnableLayer(
        input_size=input_size,
        output_size=output_size,
        node_type=LinearLUTNode,
        n=n
    )
    
    learnable_params = count_parameters(learnable_layer)
    print(f"LearnableLayer total parameters: {learnable_params:,}")
    
    # Create AdaptiveLayer with different configurations
    configs = [
        {"connection_fraction": 0.5, "num_clusters": 4, "name": "50% connections, 4 clusters"},
        {"connection_fraction": 0.3, "num_clusters": 8, "name": "30% connections, 8 clusters"},
        {"connection_fraction": 0.2, "num_clusters": 16, "name": "20% connections, 16 clusters"},
    ]
    
    print("\n" + "-" * 70)
    print("AdaptiveLayer Configurations:")
    print("-" * 70)
    
    for config in configs:
        adaptive_layer = AdaptiveLayer(
            input_size=input_size,
            output_size=output_size,
            node_type=LinearLUTNode,
            n=n,
            connection_fraction=config["connection_fraction"],
            num_clusters=config["num_clusters"]
        )
        
        adaptive_params = count_parameters(adaptive_layer)
        reduction = (1.0 - adaptive_params / learnable_params) * 100
        
        print(f"\n{config['name']}:")
        print(f"  Total parameters: {adaptive_params:,}")
        print(f"  Parameter reduction: {reduction:.1f}%")
        print(f"  Efficiency ratio: {learnable_params / adaptive_params:.2f}x")
        
        # Get detailed breakdown if available
        if hasattr(adaptive_layer, 'count_parameters'):
            breakdown = adaptive_layer.count_parameters()
            if isinstance(breakdown, dict):
                print(f"  Breakdown:")
                for key, value in breakdown.items():
                    print(f"    {key}: {value:,}")
    
    print("\n" + "=" * 70)
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 32
    x = torch.randn(batch_size, input_size)
    
    # Test LearnableLayer
    learnable_layer.eval()
    with torch.no_grad():
        y_learnable = learnable_layer(x)
    print(f"✓ LearnableLayer output shape: {y_learnable.shape}")
    
    # Test AdaptiveLayer
    adaptive_layer = AdaptiveLayer(
        input_size=input_size,
        output_size=output_size,
        node_type=LinearLUTNode,
        n=n,
        connection_fraction=0.5,
        num_clusters=4
    )
    adaptive_layer.eval()
    with torch.no_grad():
        y_adaptive = adaptive_layer(x)
    print(f"✓ AdaptiveLayer output shape: {y_adaptive.shape}")
    
    # Test training mode
    print("\nTesting training mode (soft selection)...")
    adaptive_layer.train()
    y_adaptive_train = adaptive_layer(x)
    print(f"✓ AdaptiveLayer training output shape: {y_adaptive_train.shape}")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("  • AdaptiveLayer successfully reduces parameters through:")
    print("    1. Sparse connections (each node sees only a fraction of inputs)")
    print("    2. Weight sharing (clusters of nodes share weight matrices)")
    print("  • The layer uses soft selection during training and hard selection")
    print("    during evaluation, similar to Gumbel-Softmax")
    print("  • Different configurations allow trading off between parameter")
    print("    efficiency and model capacity")
    print("=" * 70)


def demonstrate_scaling():
    """Show how parameter counts scale with different input sizes."""
    
    print("\n" + "=" * 70)
    print("Scaling Analysis")
    print("=" * 70)
    
    output_size = 128
    n = 6
    input_sizes = [256, 512, 784, 1024, 2048]
    
    print(f"\nFixed: output_size={output_size}, n={n}")
    print(f"\n{'Input Size':<12} {'Learnable':<15} {'Adaptive':<15} {'Reduction':<12}")
    print("-" * 60)
    
    for input_size in input_sizes:
        learnable = LearnableLayer(input_size, output_size, LinearLUTNode, n)
        adaptive = AdaptiveLayer(
            input_size, output_size, LinearLUTNode, n,
            connection_fraction=0.5, num_clusters=4
        )
        
        learnable_params = count_parameters(learnable)
        adaptive_params = count_parameters(adaptive)
        reduction = (1.0 - adaptive_params / learnable_params) * 100
        
        print(f"{input_size:<12} {learnable_params:<15,} {adaptive_params:<15,} {reduction:>10.1f}%")
    
    print("=" * 70)


if __name__ == "__main__":
    compare_layers()
    demonstrate_scaling()
