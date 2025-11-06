#!/usr/bin/env python3
"""
Comprehensive learning test for ConvolutionalLayer with bit flip and gradient stabilization.

Creates synthetic edge-detection dataset and trains a convolutional model with:
1. Basic functionality (forward pass)
2. Learning without augmentation
3. Learning with bit flip
4. Learning with gradient stabilization
5. Learning with both features

Usage:
    python tests/test_convolutional_learning.py
"""

import torch
import torch.nn as nn
import sys
import os

import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class SimpleConvModel(nn.Module):
    """Simple model: ConvolutionalLayer → RandomLayer → GroupSum."""

    def __init__(self, conv_layer, feedforward_layer, groupsum_layer):
        super().__init__()
        self.conv = conv_layer
        self.feedforward = feedforward_layer
        self.groupsum = groupsum_layer

    def forward(self, x):
        # Conv layer: (batch, 3, 16, 16) → (batch, 16, 14, 14)
        x = self.conv(x)

        # Flatten: (batch, 16, 14, 14) → (batch, 16*14*14)
        x = x.reshape(x.shape[0], -1)

        # Random layer: (batch, 16*14*14) → (batch, 50)
        x = self.feedforward(x)

        # GroupSum: (batch, 50) → (batch, 2)
        x = self.groupsum(x)

        return x


def create_edge_detection_dataset(num_samples, image_size=16, num_channels=3):
    """
    Create synthetic edge-detection dataset.

    Class 0: Vertical edges (left half white, right half black)
    Class 1: Horizontal edges (top half white, bottom half black)

    Args:
        num_samples: Number of samples per class
        image_size: Size of square images
        num_channels: Number of color channels (3 for RGB)

    Returns:
        images: (2*num_samples, num_channels, image_size, image_size) float tensor
        labels: (2*num_samples,) long tensor
    """
    # Class 0: Vertical edges
    images_class0 = torch.zeros(num_samples, num_channels, image_size, image_size)
    images_class0[:, :, :, :image_size//2] = 1.0  # Left half white

    labels_class0 = torch.zeros(num_samples, dtype=torch.long)

    # Class 1: Horizontal edges
    images_class1 = torch.zeros(num_samples, num_channels, image_size, image_size)
    images_class1[:, :, :image_size//2, :] = 1.0  # Top half white

    labels_class1 = torch.ones(num_samples, dtype=torch.long)

    # Combine
    images = torch.cat([images_class0, images_class1], dim=0)
    labels = torch.cat([labels_class0, labels_class1], dim=0)

    # Shuffle
    perm = torch.randperm(2 * num_samples)
    images = images[perm]
    labels = labels[perm]

    return images, labels


def train_model(model, train_images, train_labels, num_epochs=20, lr=0.01, device='cuda'):
    """Train model and return final accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if not torch.cuda.is_available() and device == 'cuda':
        print("✗ CUDA not available. Cannot train on GPU.")
        device = 'cpu'
    
    model.to(device)
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)

    model.train()
    for epoch in tqdm.tqdm(range(num_epochs), desc="Training Epochs", ncols=120, unit="epoch"):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(train_images)
        loss = criterion(outputs, train_labels)

        # Backward pass
        loss.backward()

        # Check gradients
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item()

        if grad_norm == 0.0:
            print(f"  ✗ Epoch {epoch+1}: Gradients are zero! Gradient flow is broken.")
            return 0.0

        optimizer.step()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(train_images)
        _, predicted = outputs.max(1)
        accuracy = (predicted == train_labels).float().mean()

    return accuracy.item()


def test_scenario(scenario_name, layer_config, device='cuda'):
    """Test a specific configuration scenario."""
    print(f"\n{'='*80}")
    print(f"  {scenario_name}")
    print(f"{'='*80}")

    try:
        from difflut.layers import ConvolutionConfig
        # from difflut.nodes import DWNNode
        from difflut import REGISTRY
        from difflut.nodes.node_config import NodeConfig
        from difflut.utils.modules import GroupSum
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

    # Create data
    train_images, train_labels = create_edge_detection_dataset(num_samples=50, image_size=16, num_channels=1)
    test_images, test_labels = create_edge_detection_dataset(num_samples=20, image_size=16, num_channels=1)

    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)

    # Create conv config
    conv_config = ConvolutionConfig(
        tree_depth=1,
        in_channels=1,
        out_channels=16,
        receptive_field=3,
        stride=1,
        padding=0,
        chunk_size=8,
        seed=42
    )

    node_type = REGISTRY.get_node("probabilistic")
    layer_type = REGISTRY.get_layer("random")
    conv_layer_type = REGISTRY.get_layer("convolutional")

    # Create node config
    node_config = NodeConfig(input_dim=6, output_dim=1)

    # Create convolutional layer
    conv_layer = conv_layer_type(
        convolution_config=conv_config,
        node_type=node_type,
        node_kwargs=node_config,
        layer_type=layer_type,
        n_inputs_per_node=6,
        layer_config=layer_config
    )

    # Create feedforward layer
    feedforward_layer = layer_type(
        input_size=16 * 14 * 14,
        output_size=50,
        node_type=node_type,
        node_kwargs=node_config,
        seed=43
    )

    # Create groupsum
    groupsum_layer = GroupSum(k=2, tau=10.0, use_randperm=False)

    # Build model
    model = SimpleConvModel(conv_layer, feedforward_layer, groupsum_layer).to(device)

    print(f"\nModel architecture:")
    print(f"  ConvLayer: 1→16 channels, 3x3 RF, output: (batch, 16, 14, 14)")
    print(f"  Flatten: → (batch, {16*14*14})")
    print(f"  RandomLayer: {16*14*14} → 50 nodes")
    print(f"  GroupSum: 50 → 2 classes")
    print(f"\nTraining configuration:")
    print(f"  {layer_config}")

    # Train
    print(f"\nTraining...")
    train_acc = train_model(model, train_images, train_labels, num_epochs=20, lr=0.01, device=device)

    # Test
    print(f"\nTesting...")
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_images)
        _, test_predicted = test_outputs.max(1)
        test_accuracy = (test_predicted == test_labels).float().mean()

    print(f"\nResults:")
    print(f"  Train Accuracy: {train_acc:.2%}")
    print(f"  Test Accuracy:  {test_accuracy.item():.2%}")

    # Success criterion: > 70% (well above random 50%)
    if test_accuracy > 0.70:
        print(f"\n✓ {scenario_name} PASSED")
        return True
    else:
        print(f"\n✗ {scenario_name} FAILED (accuracy {test_accuracy.item():.2%} < 70%)")
        return False


def main():
    """Run all test scenarios."""
    print("="*80)
    print("  CONVOLUTIONAL LAYER LEARNING TEST SUITE")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Using device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    # Import LayerConfig
    try:
        from difflut.layers import LayerConfig
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return 1

    # Test scenarios
    results = {}

    # Scenario 1: No augmentation (baseline)
    results['baseline'] = test_scenario(
        "Scenario 1: Baseline (No Augmentation)",
        LayerConfig(),  # Default: no flip, no grad stab
        device=device
    )

    # Scenario 2: With bit flip
    results['bit_flip'] = test_scenario(
        "Scenario 2: With Bit Flip Augmentation",
        LayerConfig(flip_probability=0.1),
        device=device
    )

    # Scenario 3: With gradient stabilization
    results['grad_stab'] = test_scenario(
        "Scenario 3: With Gradient Stabilization",
        LayerConfig(grad_stabilization='layerwise', grad_target_std=1.0),
        device=device
    )

    # Scenario 4: With both
    results['both'] = test_scenario(
        "Scenario 4: With Both Features",
        LayerConfig(
            flip_probability=0.1,
            grad_stabilization='layerwise',
            grad_target_std=1.0
        ),
        device=device
    )

    # Summary
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80)

    passed = sum(results.values())
    total = len(results)

    print(f"\nScenario 1 (Baseline):              {'✓ PASS' if results['baseline'] else '✗ FAIL'}")
    print(f"Scenario 2 (Bit Flip):              {'✓ PASS' if results['bit_flip'] else '✗ FAIL'}")
    print(f"Scenario 3 (Gradient Stabilization):{'✓ PASS' if results['grad_stab'] else '✗ FAIL'}")
    print(f"Scenario 4 (Both Features):         {'✓ PASS' if results['both'] else '✗ FAIL'}")

    print(f"\n{'='*80}")
    if passed == total:
        print(f"  ✓ ALL TESTS PASSED ({passed}/{total})")
        print(f"{'='*80}")
        print("\n🎉 ConvolutionalLayer learns successfully with all configurations!")
        print("   Bit flip and gradient stabilization are working correctly.\n")
        return 0
    else:
        print(f"  ⚠ SOME TESTS FAILED ({passed}/{total} passed)")
        print(f"{'='*80}")
        print(f"\n⚠ {total - passed} test(s) failed. Check output above for details.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
