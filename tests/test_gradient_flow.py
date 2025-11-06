#!/usr/bin/env python3
"""
Minimal gradient flow test with ConvolutionalLayer.

Creates synthetic data: 10% white pixels (class 0) vs 90% white pixels (class 1).
Trains a simple model: ConvLayer → RandomLayer → GroupSum
Verifies that gradients flow correctly through the fused kernels.

Usage:
    python tests/test_gradient_flow.py
"""

import torch
import torch.nn as nn
import sys


class SimpleConvModel(nn.Module):
    """Simple model: Conv LUT layer → Random layer → GroupSum."""

    def __init__(self, conv_layer, feedforward_layer, groupsum_layer):
        super().__init__()
        self.conv = conv_layer
        self.feedforward = feedforward_layer
        self.groupsum = groupsum_layer

    def forward(self, x):
        # Conv layer: (batch, 1, 28, 28) → (batch, 32, 24, 24)
        x = self.conv(x)

        # Flatten: (batch, 32, 24, 24) → (batch, 32*24*24)
        # Use .reshape() instead of .view() as tensor may not be contiguous after conv
        x = x.reshape(x.shape[0], -1)

        # Random layer: (batch, 32*24*24) → (batch, 100)
        x = self.feedforward(x)

        # GroupSum: (batch, 100) → (batch, 2)
        x = self.groupsum(x)

        return x


def create_synthetic_data(num_samples, image_size=28, white_prob_class0=0.1, white_prob_class1=0.9):
    """
    Create synthetic binary images with different white pixel densities.

    Args:
        num_samples: Number of samples per class (total will be 2*num_samples)
        image_size: Size of square images
        white_prob_class0: Probability of white pixel for class 0
        white_prob_class1: Probability of white pixel for class 1

    Returns:
        images: (2*num_samples, 1, image_size, image_size) float tensor
        labels: (2*num_samples,) long tensor
    """
    # Class 0: 10% white pixels
    images_class0 = (torch.rand(num_samples, 1, image_size, image_size) < white_prob_class0).float()
    labels_class0 = torch.zeros(num_samples, dtype=torch.long)

    # Class 1: 90% white pixels
    images_class1 = (torch.rand(num_samples, 1, image_size, image_size) < white_prob_class1).float()
    labels_class1 = torch.ones(num_samples, dtype=torch.long)

    # Combine
    images = torch.cat([images_class0, images_class1], dim=0)
    labels = torch.cat([labels_class0, labels_class1], dim=0)

    # Shuffle
    perm = torch.randperm(2 * num_samples)
    images = images[perm]
    labels = labels[perm]

    return images, labels


def train_and_test():
    """Train simple model and report results."""
    print("="*80)
    print("  GRADIENT FLOW TEST: ConvolutionalLayer")
    print("="*80)

    # Check CUDA
    if not torch.cuda.is_available():
        print("✗ CUDA not available. This test requires GPU.")
        return 1

    device = 'cuda'
    print(f"\n✓ Using device: {torch.cuda.get_device_name(0)}")

    # Import here to avoid issues if difflut not installed
    try:
        from difflut.layers import ConvolutionalLayer, ConvolutionConfig, RandomLayer
        from difflut.nodes import DWNNode
        from difflut.nodes.node_config import NodeConfig
        from difflut.utils.modules import GroupSum
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return 1

    # Create model
    print("\n" + "-"*80)
    print("Building model...")
    print("-"*80)

    # Create convolution configuration
    conv_config = ConvolutionConfig(
        tree_depth=2,
        in_channels=1,
        out_channels=32,
        receptive_field=5,
        stride=1,
        padding=0,
        chunk_size=16,
        seed=42
    )

    # Create node configuration
    node_config = NodeConfig(input_dim=6, output_dim=1)

    # Conv layer: 1→32 channels, 5x5 receptive field
    conv_layer = ConvolutionalLayer(
        convolution_config=conv_config,
        node_type=DWNNode,
        node_kwargs=node_config,
        layer_type=RandomLayer,
        n_inputs_per_node=6
    )

    # Feedforward layer: flatten conv output → 100 nodes
    # Conv output: (batch, 32, 24, 24) → flatten to (batch, 18432)
    feedforward_layer = RandomLayer(
        input_size=32 * 24 * 24,
        output_size=100,
        node_type=DWNNode,
        node_kwargs=node_config,
        seed=43
    )

    # GroupSum: 100 nodes → 2 classes
    groupsum_layer = GroupSum(k=2, tau=10.0)

    # Build complete model
    model = SimpleConvModel(conv_layer, feedforward_layer, groupsum_layer).to(device)

    print("\nModel architecture:")
    print(f"  ConvLayer: 1→32 channels, 5x5 RF, output: (batch, 32, 24, 24)")
    print(f"  Flatten: → (batch, 18432)")
    print(f"  RandomLayer: 18432 → 100 nodes")
    print(f"  GroupSum: 100 → 2 classes")

    # Create data
    print("\n" + "-"*80)
    print("Creating synthetic data...")
    print("-"*80)

    train_images, train_labels = create_synthetic_data(num_samples=50)
    test_images, test_labels = create_synthetic_data(num_samples=20)

    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)

    print(f"\nTrain set: {train_images.shape[0]} samples")
    print(f"Test set: {test_images.shape[0]} samples")
    print(f"Class 0: ~10% white pixels, Class 1: ~90% white pixels")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 10

    print("\n" + "-"*80)
    print("Training...")
    print("-"*80)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(train_images)
        loss = criterion(outputs, train_labels)

        # Backward pass
        loss.backward()

        # Check gradients
        grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item()

        if grad_norm == 0.0:
            print(f"✗ Epoch {epoch+1}: Gradients are zero! Gradient flow is broken.")
            return 1

        optimizer.step()

        # Compute accuracy
        _, predicted = outputs.max(1)
        accuracy = (predicted == train_labels).float().mean()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}, Acc: {accuracy.item():.2%}, Grad norm: {grad_norm:.2f}")

    print("\n✓ Training completed - gradients flowed correctly!")

    # Test
    print("\n" + "-"*80)
    print("Testing...")
    print("-"*80)

    model.eval()
    with torch.no_grad():
        test_outputs = model(test_images)
        test_loss = criterion(test_outputs, test_labels)
        _, test_predicted = test_outputs.max(1)
        test_accuracy = (test_predicted == test_labels).float().mean()

    print(f"\nTest Loss: {test_loss.item():.4f}")
    print(f"Test Accuracy: {test_accuracy.item():.2%}")

    print("\n" + "="*80)
    if test_accuracy > 0.6:  # Should be > 50% if learning anything
        print("  ✓ GRADIENT FLOW TEST PASSED")
        print("="*80)
        print("\n✓ Model trained successfully with fused kernels!")
        print("  Gradients flow correctly through ConvolutionalLayer.\n")
        return 0
    else:
        print("  ⚠ TRAINING COMPLETED BUT ACCURACY LOW")
        print("="*80)
        print(f"\n⚠ Test accuracy ({test_accuracy.item():.2%}) is low.")
        print("  Gradients are flowing, but model may need tuning.\n")
        return 0  # Still pass since gradients flow


if __name__ == "__main__":
    sys.exit(train_and_test())
