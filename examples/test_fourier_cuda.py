"""
Test script for FourierNode CUDA implementation.
"""
import torch
import numpy as np
from difflut.nodes.fourier_node import FourierNode

def test_fourier_node_cuda():
    """Test FourierNode with CUDA acceleration."""
    
    print("=" * 80)
    print("Testing FourierNode CUDA Implementation")
    print("=" * 80)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")
    
    # Test parameters
    num_inputs = 4
    output_dim = 2
    batch_size = 16
    
    # Create node with CUDA enabled
    print(f"\nCreating FourierNode with {num_inputs} inputs, {output_dim} outputs")
    node = FourierNode(
        num_inputs=num_inputs,
        output_dim=output_dim,
        use_all_frequencies=True,
        max_amplitude=0.5,
        use_cuda=True
    )
    
    print(f"Node uses CUDA: {node.use_cuda}")
    print(f"Number of frequencies: {node.num_frequencies}")
    
    # Test on CPU
    print("\n" + "-" * 80)
    print("Testing on CPU")
    print("-" * 80)
    
    x_cpu = torch.randn(batch_size, num_inputs)
    print(f"Input shape: {x_cpu.shape}")
    
    # Forward train
    node.train()
    output_train = node.forward_train(x_cpu)
    print(f"Output train shape: {output_train.shape}")
    print(f"Output train range: [{output_train.min().item():.4f}, {output_train.max().item():.4f}]")
    
    # Forward eval
    node.eval()
    output_eval = node.forward_eval(x_cpu)
    print(f"Output eval shape: {output_eval.shape}")
    print(f"Output eval range: [{output_eval.min().item():.4f}, {output_eval.max().item():.4f}]")
    
    # Test backward
    node.train()
    x_cpu.requires_grad = True
    output = node.forward_train(x_cpu)
    loss = output.sum()
    loss.backward()
    print(f"Gradient computed: {x_cpu.grad is not None}")
    print(f"Gradient shape: {x_cpu.grad.shape}")
    print(f"Gradient norm: {x_cpu.grad.norm().item():.6f}")
    
    # Test on GPU if available
    if cuda_available:
        print("\n" + "-" * 80)
        print("Testing on GPU")
        print("-" * 80)
        
        node_gpu = node.cuda()
        x_gpu = x_cpu.detach().cuda()
        print(f"Input device: {x_gpu.device}")
        
        # Forward train
        node_gpu.train()
        output_train_gpu = node_gpu.forward_train(x_gpu)
        print(f"Output train shape: {output_train_gpu.shape}")
        print(f"Output train device: {output_train_gpu.device}")
        print(f"Output train range: [{output_train_gpu.min().item():.4f}, {output_train_gpu.max().item():.4f}]")
        
        # Forward eval
        node_gpu.eval()
        output_eval_gpu = node_gpu.forward_eval(x_gpu)
        print(f"Output eval shape: {output_eval_gpu.shape}")
        print(f"Output eval device: {output_eval_gpu.device}")
        print(f"Output eval range: [{output_eval_gpu.min().item():.4f}, {output_eval_gpu.max().item():.4f}]")
        
        # Test backward
        node_gpu.train()
        x_gpu_grad = x_gpu.detach().requires_grad_(True)
        output_gpu = node_gpu.forward_train(x_gpu_grad)
        loss_gpu = output_gpu.sum()
        loss_gpu.backward()
        print(f"Gradient computed: {x_gpu_grad.grad is not None}")
        print(f"Gradient shape: {x_gpu_grad.grad.shape}")
        print(f"Gradient device: {x_gpu_grad.grad.device}")
        print(f"Gradient norm: {x_gpu_grad.grad.norm().item():.6f}")
        
        # Compare CPU and GPU outputs
        print("\n" + "-" * 80)
        print("Comparing CPU and GPU outputs")
        print("-" * 80)
        
        output_train_cpu_moved = output_train.detach()
        output_train_gpu_moved = output_train_gpu.cpu().detach()
        
        diff = torch.abs(output_train_cpu_moved - output_train_gpu_moved)
        print(f"Max absolute difference (train): {diff.max().item():.6e}")
        print(f"Mean absolute difference (train): {diff.mean().item():.6e}")
        
        # Note: eval outputs may differ due to Heaviside discretization
        output_eval_cpu_moved = output_eval.detach()
        output_eval_gpu_moved = output_eval_gpu.cpu().detach()
        
        diff_eval = torch.abs(output_eval_cpu_moved - output_eval_gpu_moved)
        print(f"Max absolute difference (eval): {diff_eval.max().item():.6e}")
        print(f"Mean absolute difference (eval): {diff_eval.mean().item():.6e}")
        
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


def test_dominant_frequencies():
    """Test the dominant frequencies method."""
    
    print("\n" + "=" * 80)
    print("Testing Dominant Frequencies")
    print("=" * 80)
    
    num_inputs = 3
    node = FourierNode(num_inputs=num_inputs, output_dim=1, use_cuda=True)
    
    # Get dominant frequencies
    top_freqs = node.get_dominant_frequencies(top_k=5)
    print(f"\nTop 5 dominant frequencies:")
    for i, freq in enumerate(top_freqs):
        print(f"  {i+1}. {freq.cpu().numpy()}")


def test_regularization():
    """Test regularization."""
    
    print("\n" + "=" * 80)
    print("Testing Regularization")
    print("=" * 80)
    
    node = FourierNode(num_inputs=4, output_dim=2, use_cuda=True)
    
    reg = node.regularization_loss()
    print(f"\nRegularization loss: {reg.item():.6f}")


if __name__ == "__main__":
    test_fourier_node_cuda()
    test_dominant_frequencies()
    test_regularization()
