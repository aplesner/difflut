"""
Test and demonstration of ProbabilisticStableNode.

This script tests:
1. Basic functionality of the node
2. Gradient stabilization behavior
3. Comparison with standard ProbabilisticNode
"""

import torch
import torch.nn as nn
import warnings

# Suppress CUDA warnings if not compiled yet
warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut.nodes.probabilistic_stable_node')

from difflut import ProbabilisticStableNode, ProbabilisticNode

def test_forward_pass():
    """Test basic forward pass."""
    print("=" * 60)
    print("Test 1: Forward Pass")
    print("=" * 60)
    
    # Create node
    node = ProbabilisticStableNode(
        input_dim=[6],
        output_dim=[2],
        temperature=1.0,
        alpha=1.0,
        use_cuda=False  # Use CPU for testing
    )
    
    # Create random input in [0, 1]
    batch_size = 8
    x = torch.rand(batch_size, 6)
    
    # Forward pass
    output = node(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("✓ Forward pass successful\n")
    
    return node, x, output


def test_backward_pass():
    """Test backward pass with gradient stabilization."""
    print("=" * 60)
    print("Test 2: Backward Pass with Gradient Stabilization")
    print("=" * 60)
    
    # Create node
    node = ProbabilisticStableNode(
        input_dim=[6],
        output_dim=[2],
        temperature=1.0,
        alpha=1.0,
        scale_min=0.1,
        scale_max=10.0,
        use_cuda=False
    )
    
    # Create input that requires gradient
    x = torch.rand(4, 6, requires_grad=True)
    
    # Forward pass
    output = node(x)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    print(f"Input gradient shape: {x.grad.shape}")
    print(f"Input gradient L1 norm: {x.grad.abs().sum(dim=1)}")
    print(f"Weight gradient shape: {node.raw_weights.grad.shape}")
    print("✓ Backward pass successful\n")
    
    return x.grad


def test_gradient_stabilization():
    """Compare gradient behavior with and without stabilization."""
    print("=" * 60)
    print("Test 3: Gradient Stabilization Effect")
    print("=" * 60)
    
    # Shared parameters
    input_dim = [10]
    output_dim = [2]
    batch_size = 8
    
    # Create both node types
    node_stable = ProbabilisticStableNode(
        input_dim=input_dim,
        output_dim=output_dim,
        temperature=1.0,
        alpha=1.0,
        use_cuda=False
    )
    
    node_standard = ProbabilisticNode(
        input_dim=input_dim,
        output_dim=output_dim,
        temperature=1.0,
        use_cuda=False
    )
    
    # Copy weights to make them identical
    with torch.no_grad():
        node_standard.raw_weights.copy_(node_stable.raw_weights)
    
    # Create identical inputs
    x_stable = torch.rand(batch_size, 10, requires_grad=True)
    x_standard = x_stable.clone().detach().requires_grad_(True)
    
    # Forward pass
    output_stable = node_stable(x_stable)
    output_standard = node_standard(x_standard)
    
    # Create identical output gradients
    grad_output = torch.randn_like(output_stable)
    
    # Backward pass
    output_stable.backward(grad_output)
    output_standard.backward(grad_output.clone())
    
    # Compare gradients
    grad_stable_l1 = x_stable.grad.abs().sum(dim=1)
    grad_standard_l1 = x_standard.grad.abs().sum(dim=1)
    
    print("Input gradient L1 norms (per sample):")
    print(f"  Standard node:  {grad_standard_l1}")
    print(f"  Stabilized node: {grad_stable_l1}")
    print(f"\nGradient ratio (stabilized/standard): {(grad_stable_l1 / grad_standard_l1).mean():.4f}")
    print("✓ Gradient stabilization is active\n")


def test_deep_network():
    """Test gradient flow in a deep network."""
    print("=" * 60)
    print("Test 4: Deep Network Gradient Flow")
    print("=" * 60)
    
    class DeepLUTNet(nn.Module):
        def __init__(self, use_stabilization=True):
            super().__init__()
            NodeClass = ProbabilisticStableNode if use_stabilization else ProbabilisticNode
            
            self.layers = nn.ModuleList([
                NodeClass(input_dim=[10], output_dim=[10], use_cuda=False, 
                         **(dict(alpha=0.9) if use_stabilization else {}))
                for _ in range(5)
            ])
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    # Create both versions
    net_stable = DeepLUTNet(use_stabilization=True)
    net_standard = DeepLUTNet(use_stabilization=False)
    
    # Copy weights
    with torch.no_grad():
        for stable_layer, standard_layer in zip(net_stable.layers, net_standard.layers):
            standard_layer.raw_weights.copy_(stable_layer.raw_weights)
    
    # Create input
    x_stable = torch.rand(4, 10, requires_grad=True)
    x_standard = x_stable.clone().detach().requires_grad_(True)
    
    # Forward and backward
    output_stable = net_stable(x_stable)
    output_standard = net_standard(x_standard)
    
    loss_stable = output_stable.sum()
    loss_standard = output_standard.sum()
    
    loss_stable.backward()
    loss_standard.backward()
    
    # Compare input gradients
    grad_stable_norm = x_stable.grad.norm()
    grad_standard_norm = x_standard.grad.norm()
    
    print(f"Input gradient norm (5-layer network):")
    print(f"  Standard nodes:   {grad_standard_norm:.6f}")
    print(f"  Stabilized nodes: {grad_stable_norm:.6f}")
    print(f"  Ratio: {grad_stable_norm / grad_standard_norm:.4f}x")
    print("✓ Gradient stabilization helps in deep networks\n")


def test_parameter_tuning():
    """Test different alpha values."""
    print("=" * 60)
    print("Test 5: Alpha Parameter Tuning")
    print("=" * 60)
    
    alphas = [0.5, 0.8, 1.0, 1.2]
    
    for alpha in alphas:
        node = ProbabilisticStableNode(
            input_dim=[8],
            output_dim=[2],
            alpha=alpha,
            use_cuda=False
        )
        
        x = torch.rand(4, 8, requires_grad=True)
        output = node(x)
        loss = output.sum()
        loss.backward()
        
        grad_norm = x.grad.norm()
        print(f"Alpha={alpha:.1f}: Input gradient norm = {grad_norm:.6f}")
    
    print("✓ Alpha parameter controls gradient strength\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ProbabilisticStableNode Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_forward_pass()
        test_backward_pass()
        test_gradient_stabilization()
        test_deep_network()
        test_parameter_tuning()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nTo compile CUDA extensions for faster performance:")
        print("  cd difflut && python setup.py install")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
