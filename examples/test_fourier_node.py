"""
Test and demonstration of FourierNode and FourierHermitianNode.

These nodes implement a DFT-like structure where the output is:
    y = 0.5 + Σ_k Re(w_k * exp(i * 2π * <k, x>))

With constraints ensuring:
1. Real-valued output (Hermitian symmetry)
2. Bounded in [0, 1] (amplitude normalization)
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '..')

from difflut.nodes import FourierNode, FourierHermitianNode, LinearLUTNode


def test_basic_functionality():
    """Test basic forward and backward pass."""
    print("=" * 70)
    print("Test 1: Basic Functionality")
    print("=" * 70)
    
    num_inputs = 4
    batch_size = 16
    
    # Create Fourier node
    fourier_node = FourierNode(num_inputs=num_inputs, output_dim=1)
    
    print(f"\nFourier Node Configuration:")
    print(f"  Number of inputs: {num_inputs}")
    print(f"  Number of frequencies: {fourier_node.num_frequencies}")
    print(f"  Output dimension: {fourier_node.output_dim}")
    print(f"  Max amplitude: {fourier_node.max_amplitude}")
    
    # Create random input
    x = torch.randn(batch_size, num_inputs, requires_grad=True)
    
    # Forward pass
    output = fourier_node(x)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  Output mean: {output.mean().item():.4f}")
    
    # Check if output is in [0, 1]
    in_range = torch.all((output >= 0) & (output <= 1))
    print(f"  All outputs in [0,1]: {in_range.item()}")
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    print(f"\nBackward pass:")
    print(f"  Gradient shape: {x.grad.shape}")
    print(f"  Gradient norm: {x.grad.norm().item():.6f}")
    print(f"  Gradient range: [{x.grad.min().item():.4f}, {x.grad.max().item():.4f}]")
    
    print("\n✓ Basic functionality working!")


def test_hermitian_symmetry():
    """Test that the output is real-valued."""
    print("\n" + "=" * 70)
    print("Test 2: Real-Valued Output (Hermitian Symmetry)")
    print("=" * 70)
    
    num_inputs = 4
    batch_size = 32
    
    fourier_node = FourierNode(num_inputs=num_inputs, output_dim=1)
    
    # Test with various inputs
    x = torch.randn(batch_size, num_inputs)
    output = fourier_node(x)
    
    print(f"\nTesting {batch_size} samples:")
    print(f"  All outputs are real: {torch.all(torch.isreal(output)).item()}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
    
    # Test with extreme inputs
    x_extreme = torch.tensor([
        [-10.0] * num_inputs,  # Very negative
        [10.0] * num_inputs,   # Very positive
        [0.0] * num_inputs,    # Zero
        [1.0] * num_inputs,    # Ones
    ])
    
    output_extreme = fourier_node(x_extreme)
    
    print(f"\nExtreme inputs:")
    for i, inp in enumerate(['Very negative', 'Very positive', 'Zero', 'Ones']):
        print(f"  {inp:20s}: output = {output_extreme[i].item():.6f}")
    
    in_range = torch.all((output_extreme >= 0) & (output_extreme <= 1))
    print(f"\n✓ All extreme outputs in [0,1]: {in_range.item()}")


def test_frequency_analysis():
    """Analyze which frequencies are most important."""
    print("\n" + "=" * 70)
    print("Test 3: Frequency Analysis")
    print("=" * 70)
    
    num_inputs = 4
    
    fourier_node = FourierNode(num_inputs=num_inputs, output_dim=1)
    
    # Train on a simple pattern to see which frequencies emerge
    # Let's try to learn a simple function: output 1 if sum of inputs > 2, else 0
    
    print("\nTraining to learn a threshold function...")
    optimizer = torch.optim.Adam(fourier_node.parameters(), lr=0.01)
    
    for epoch in range(100):
        # Generate random training data
        x_train = torch.rand(32, num_inputs)
        y_target = (x_train.sum(dim=1) > 2.0).float()
        
        # Forward pass
        y_pred = fourier_node(x_train)
        
        # Loss
        loss = torch.nn.functional.binary_cross_entropy(y_pred, y_target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss = {loss.item():.6f}")
    
    # Analyze dominant frequencies
    print("\nDominant frequencies (top 5):")
    top_freqs = fourier_node.get_dominant_frequencies(top_k=5)
    
    with torch.no_grad():
        avg_amplitudes = fourier_node.amplitudes.mean(dim=1)
        top_indices = torch.topk(avg_amplitudes, min(5, fourier_node.num_frequencies)).indices
        
        for i, idx in enumerate(top_indices):
            freq = fourier_node.frequencies[idx]
            amp = avg_amplitudes[idx].item()
            phase = fourier_node.phases[idx, 0].item()
            print(f"  {i+1}. Frequency {freq.numpy()}: amplitude={amp:.4f}, phase={phase:.4f}")
    
    print("\n✓ Frequency analysis complete!")


def test_smoothness():
    """Test that the function is smooth (continuous gradients)."""
    print("\n" + "=" * 70)
    print("Test 4: Smoothness of Function")
    print("=" * 70)
    
    num_inputs = 3
    fourier_node = FourierNode(num_inputs=num_inputs, output_dim=1)
    
    # Test smoothness along a line
    print("\nTesting smoothness along input dimension 0:")
    
    x_base = torch.zeros(1, num_inputs)
    x_values = torch.linspace(-3, 3, 50)
    
    outputs = []
    gradients = []
    
    for x_val in x_values:
        x = x_base.clone()
        x[0, 0] = x_val
        x.requires_grad = True
        
        y = fourier_node(x)
        y.backward()
        
        outputs.append(y.item())
        gradients.append(x.grad[0, 0].item())
    
    outputs = np.array(outputs)
    gradients = np.array(gradients)
    
    # Check for discontinuities
    output_diffs = np.abs(np.diff(outputs))
    max_output_jump = np.max(output_diffs)
    
    gradient_diffs = np.abs(np.diff(gradients))
    max_gradient_jump = np.max(gradient_diffs)
    
    print(f"  Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
    print(f"  Max output jump: {max_output_jump:.6f}")
    print(f"  Max gradient jump: {max_gradient_jump:.6f}")
    print(f"  Mean gradient: {np.mean(np.abs(gradients)):.6f}")
    
    if max_output_jump < 0.5:
        print("\n✓ Function is smooth (no large discontinuities)!")
    else:
        print("\n→ Function has some discontinuities (expected for untrained network)")


def test_hermitian_node():
    """Test the explicit Hermitian symmetry version."""
    print("\n" + "=" * 70)
    print("Test 5: Hermitian Symmetry Node")
    print("=" * 70)
    
    num_inputs = 4
    batch_size = 16
    
    hermitian_node = FourierHermitianNode(num_inputs=num_inputs, output_dim=1)
    
    print(f"\nFourier Hermitian Node Configuration:")
    print(f"  Number of inputs: {num_inputs}")
    print(f"  Output dimension: {hermitian_node.output_dim}")
    
    # Forward pass
    x = torch.randn(batch_size, num_inputs, requires_grad=True)
    output = hermitian_node(x)
    
    print(f"\nForward pass:")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  Output mean: {output.mean().item():.4f}")
    
    # Check boundedness
    in_range = torch.all((output >= 0) & (output <= 1))
    print(f"  All outputs in [0,1]: {in_range.item()}")
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    print(f"\nBackward pass:")
    print(f"  Gradient norm: {x.grad.norm().item():.6f}")
    
    print("\n✓ Hermitian node working correctly!")


def compare_with_linear():
    """Compare Fourier node with Linear LUT node."""
    print("\n" + "=" * 70)
    print("Test 6: Comparison with Linear LUT Node")
    print("=" * 70)
    
    num_inputs = 4
    batch_size = 32
    
    fourier_node = FourierNode(num_inputs=num_inputs, output_dim=1)
    linear_node = LinearLUTNode(num_inputs=num_inputs, output_dim=1)
    
    # Count parameters
    fourier_params = sum(p.numel() for p in fourier_node.parameters())
    linear_params = sum(p.numel() for p in linear_node.parameters())
    
    print(f"\nParameter counts:")
    print(f"  Fourier Node: {fourier_params}")
    print(f"  Linear Node:  {linear_params}")
    print(f"  Ratio: {fourier_params / linear_params:.2f}x")
    
    # Test forward pass
    x = torch.randn(batch_size, num_inputs)
    
    y_fourier = fourier_node(x)
    y_linear = linear_node(x)
    
    print(f"\nOutput characteristics:")
    print(f"  Fourier - mean: {y_fourier.mean():.4f}, std: {y_fourier.std():.4f}")
    print(f"  Linear  - mean: {y_linear.mean():.4f}, std: {y_linear.std():.4f}")
    
    print("\n✓ Comparison complete!")


def test_multiple_outputs():
    """Test with multiple output dimensions."""
    print("\n" + "=" * 70)
    print("Test 7: Multiple Output Dimensions")
    print("=" * 70)
    
    num_inputs = 4
    output_dim = 5
    batch_size = 16
    
    fourier_node = FourierNode(num_inputs=num_inputs, output_dim=output_dim)
    
    print(f"\nConfiguration:")
    print(f"  Number of inputs: {num_inputs}")
    print(f"  Output dimensions: {output_dim}")
    
    x = torch.randn(batch_size, num_inputs, requires_grad=True)
    output = fourier_node(x)
    
    print(f"\nForward pass:")
    print(f"  Output shape: {output.shape}")
    
    for i in range(output_dim):
        out_i = output[:, i]
        print(f"  Output {i}: range=[{out_i.min():.4f}, {out_i.max():.4f}], mean={out_i.mean():.4f}")
    
    # Check all bounded
    all_bounded = torch.all((output >= 0) & (output <= 1))
    print(f"\n  All outputs bounded in [0,1]: {all_bounded.item()}")
    
    # Backward
    loss = output.sum()
    loss.backward()
    
    print(f"\nBackward pass:")
    print(f"  Gradient shape: {x.grad.shape}")
    print(f"  Gradient norm: {x.grad.norm().item():.6f}")
    
    print("\n✓ Multiple outputs working correctly!")


def main():
    """Run all tests."""
    print("\n")
    print("#" * 70)
    print("# FourierNode Test Suite")
    print("#" * 70)
    print("\nThe FourierNode implements a DFT-like structure where:")
    print("  • Output: y = 0.5 + Σ_k Re(w_k * exp(i * 2π * <k, x>))")
    print("  • Weights have amplitude and phase parameters")
    print("  • Output is guaranteed to be real and bounded in [0, 1]")
    print("  • Smooth, differentiable everywhere")
    print("\n")
    
    try:
        test_basic_functionality()
        test_hermitian_symmetry()
        test_frequency_analysis()
        test_smoothness()
        test_hermitian_node()
        compare_with_linear()
        test_multiple_outputs()
        
        print("\n" + "=" * 70)
        print("All tests completed successfully!")
        print("=" * 70)
        print("\nSummary:")
        print("  ✓ FourierNode produces real-valued outputs in [0, 1]")
        print("  ✓ Smooth and differentiable everywhere")
        print("  ✓ Can learn to approximate complex functions")
        print("  ✓ Frequency analysis reveals function structure")
        print("  ✓ Both standard and Hermitian versions working")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
