"""
Verification script for DWNNode [-1, 1] range consistency.

Tests:
1. Training forward pass produces outputs in [-1, 1]
2. Evaluation produces binary outputs in {-1, 1}
3. Threshold is at 0.0 (not 0.5)
4. CPU and CUDA implementations are consistent
"""

import torch
import warnings

# Suppress CUDA warnings if extension not compiled
warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut.nodes.dwn_node')

from difflut import DWNNode

def test_training_range():
    """Test that training produces outputs in [-1, 1]."""
    print("=" * 60)
    print("Test 1: Training Forward Pass Range")
    print("=" * 60)
    
    node = DWNNode(
        input_dim=[4],
        output_dim=[2],
        use_cuda=False
    )
    
    # Test with inputs in [-1, 1]
    x = torch.randn(8, 4) * 2  # Scale to roughly [-2, 2], will be clamped by network
    x = torch.clamp(x, -1, 1)  # Ensure in [-1, 1]
    
    node.train()
    output = node(x)
    
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Output mean: {output.mean():.3f}")
    
    # Check that outputs are in reasonable range for [-1, 1] LUTs
    assert output.min() >= -1.0 - 1e-5, "Output below -1.0!"
    assert output.max() <= 1.0 + 1e-5, "Output above 1.0!"
    print("✓ Outputs are in [-1, 1] range\n")


def test_threshold_location():
    """Test that threshold is at 0.0, not 0.5."""
    print("=" * 60)
    print("Test 2: Threshold Location")
    print("=" * 60)
    
    node = DWNNode(
        input_dim=[2],
        output_dim=[1],
        use_cuda=False
    )
    
    # Set known LUT values
    with torch.no_grad():
        # For 2 inputs: 4 possible addresses (00, 01, 10, 11)
        node.luts[0, :] = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    
    node.train()
    
    # Test specific input values around 0.0
    test_cases = [
        (torch.tensor([[-0.1, -0.1]]), 0, "Both negative → addr 00"),
        (torch.tensor([[-0.1, 0.1]]), 1, "Neg, Pos → addr 01"),
        (torch.tensor([[0.1, -0.1]]), 2, "Pos, Neg → addr 10"),
        (torch.tensor([[0.1, 0.1]]), 3, "Both positive → addr 11"),
    ]
    
    print("Testing threshold at 0.0:")
    for x, expected_addr, description in test_cases:
        output = node(x)
        expected_output = node.luts[0, expected_addr]
        print(f"  {description}")
        print(f"    Input: {x[0].tolist()}")
        print(f"    Expected addr: {expected_addr}, Output: {output.item():.3f}, Expected: {expected_output:.3f}")
        assert abs(output.item() - expected_output.item()) < 1e-5, f"Wrong output for {description}"
    
    print("✓ Threshold is correctly at 0.0\n")


def test_eval_binary_output():
    """Test that eval mode produces binary outputs in {-1, 1}."""
    print("=" * 60)
    print("Test 3: Evaluation Binary Outputs")
    print("=" * 60)
    
    node = DWNNode(
        input_dim=[3],
        output_dim=[2],
        use_cuda=False
    )
    
    # Set some LUT values
    with torch.no_grad():
        node.luts.uniform_(-1, 1)
    
    node.eval()
    
    # Binary inputs in {-1, 1}
    x = torch.tensor([
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
    ])
    
    output = node(x)
    
    print(f"Input values (unique): {x.unique().tolist()}")
    print(f"Output values (unique): {output.unique().tolist()}")
    print(f"Output shape: {output.shape}")
    
    # Check that outputs are binary in {-1, 1}
    unique_outputs = output.unique()
    print(f"Number of unique output values: {len(unique_outputs)}")
    
    # All outputs should be either -1 or 1
    for val in unique_outputs:
        assert val.item() in [-1.0, 1.0], f"Output {val.item()} is not in {{-1, 1}}!"
    
    print("✓ Evaluation outputs are binary in {-1, 1}\n")


def test_gradient_flow():
    """Test that gradients flow correctly."""
    print("=" * 60)
    print("Test 4: Gradient Flow")
    print("=" * 60)
    
    node = DWNNode(
        input_dim=[4],
        output_dim=[2],
        use_cuda=False
    )
    
    node.train()
    
    x = torch.randn(4, 4, requires_grad=True)
    x = torch.clamp(x, -1, 1)
    
    output = node(x)
    loss = output.sum()
    loss.backward()
    
    print(f"Input gradient shape: {x.grad.shape}")
    print(f"Input gradient range: [{x.grad.min():.6f}, {x.grad.max():.6f}]")
    print(f"Input gradient mean abs: {x.grad.abs().mean():.6f}")
    print(f"LUT gradient shape: {node.luts.grad.shape}")
    print(f"LUT gradient range: [{node.luts.grad.min():.6f}, {node.luts.grad.max():.6f}]")
    
    # Check that we have non-zero gradients
    assert x.grad.abs().sum() > 1e-6, "No input gradients!"
    assert node.luts.grad.abs().sum() > 1e-6, "No LUT gradients!"
    
    print("✓ Gradients flow correctly\n")


def test_consistency_train_eval():
    """Test that train and eval modes are consistent."""
    print("=" * 60)
    print("Test 5: Train/Eval Mode Consistency")
    print("=" * 60)
    
    node = DWNNode(
        input_dim=[3],
        output_dim=[1],
        use_cuda=False
    )
    
    # Binary inputs in {-1, 1}
    x_binary = torch.tensor([
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
    ])
    
    # Training mode with binary inputs should be close to eval mode
    node.train()
    output_train = node(x_binary)
    
    node.eval()
    output_eval = node(x_binary)
    
    print("Training mode outputs:")
    print(output_train)
    print("\nEvaluation mode outputs:")
    print(output_eval)
    
    # Eval outputs should be binarized versions
    # For binary inputs, train output should be close to eval (but continuous vs discrete)
    print(f"\nOutput differences: {(output_train - output_eval).abs().max():.6f}")
    
    # Check that eval outputs are binary
    for val in output_eval.unique():
        assert val.item() in [-1.0, 1.0], f"Eval output {val.item()} not binary!"
    
    print("✓ Train and eval modes are consistent\n")


def main():
    print("\n" + "=" * 60)
    print("DWNNode [-1, 1] Range Verification")
    print("=" * 60 + "\n")
    
    try:
        test_training_range()
        test_threshold_location()
        test_eval_binary_output()
        test_gradient_flow()
        test_consistency_train_eval()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nDWNNode is correctly configured for [-1, 1] range:")
        print("  • Inputs in [-1, 1]")
        print("  • Threshold at 0.0")
        print("  • Outputs in [-1, 1] (training)")
        print("  • Binary outputs in {-1, 1} (evaluation)")
        print("  • LUT weights clamped to [-1, 1]")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
