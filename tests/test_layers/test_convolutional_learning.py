"""
Comprehensive learning test for ConvolutionalLayer with bit flip and gradient stabilization.

Creates synthetic edge-detection dataset and trains a convolutional model with:
1. Basic functionality (forward pass)
2. Learning without augmentation
3. Learning with bit flip
4. Learning with gradient stabilization
5. Learning with both features
"""

import pytest
import torch
import torch.nn as nn
from .testing_utils import is_cuda_available


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
    images_class0[:, :, :, : image_size // 2] = 1.0  # Left half white

    labels_class0 = torch.zeros(num_samples, dtype=torch.long)

    # Class 1: Horizontal edges
    images_class1 = torch.zeros(num_samples, num_channels, image_size, image_size)
    images_class1[:, :, : image_size // 2, :] = 1.0  # Top half white

    labels_class1 = torch.ones(num_samples, dtype=torch.long)

    # Combine
    images = torch.cat([images_class0, images_class1], dim=0)
    labels = torch.cat([labels_class0, labels_class1], dim=0)

    # Shuffle
    perm = torch.randperm(2 * num_samples)
    images = images[perm]
    labels = labels[perm]

    return images, labels


def train_model(model, train_images, train_labels, num_epochs=5, lr=0.01, device="cuda"):
    """Train model and return final accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)

    model.train()
    for epoch in range(num_epochs):
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

        assert grad_norm > 0.0, f"Epoch {epoch+1}: Gradients are zero! Gradient flow is broken."

        optimizer.step()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(train_images)
        _, predicted = outputs.max(1)
        accuracy = (predicted == train_labels).float().mean()

    return accuracy.item()


# ============================================================================
# Test Scenarios Configuration
# ============================================================================


@pytest.fixture
def device():
    """Get available device for testing."""
    if not is_cuda_available():
        pytest.skip("CUDA not available")
    return "cuda"


@pytest.fixture
def train_test_data(device):
    """Create training and testing datasets."""
    train_images, train_labels = create_edge_detection_dataset(
        num_samples=50, image_size=16, num_channels=1
    )
    test_images, test_labels = create_edge_detection_dataset(
        num_samples=20, image_size=16, num_channels=1
    )

    return (
        train_images.to(device),
        train_labels.to(device),
        test_images.to(device),
        test_labels.to(device),
    )


# ============================================================================
# Parametrized Tests
# ============================================================================


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "scenario_name,layer_config",
    [
        ("baseline", None),  # Will use LayerConfig() default
        ("bit_flip", {"flip_probability": 0.1}),
        ("grad_stabilization", {"grad_stabilization": "layerwise", "grad_target_std": 1.0}),
        (
            "both_features",
            {"flip_probability": 0.1, "grad_stabilization": "layerwise", "grad_target_std": 1.0},
        ),
    ],
)
def test_convolutional_learning_scenarios(scenario_name, layer_config, device, train_test_data):
    """Test ConvolutionalLayer learning with different configurations."""
    from difflut import REGISTRY
    from difflut.layers import ConvolutionConfig, LayerConfig
    from difflut.nodes.node_config import NodeConfig
    from difflut.utils.modules import GroupSum

    train_images, train_labels, test_images, test_labels = train_test_data

    # Create layer config
    if layer_config is None:
        layer_cfg = LayerConfig()
    else:
        layer_cfg = LayerConfig(**layer_config)

    # Create conv config
    conv_config = ConvolutionConfig(
        tree_depth=1,
        in_channels=1,
        out_channels=16,
        receptive_field=3,
        stride=1,
        padding=0,
        chunk_size=8,
        seed=42,
    )

    node_type = REGISTRY.get_node("probabilistic")
    layer_type = REGISTRY.get_layer("random")
    conv_layer_type = REGISTRY.get_convolutional_layer("convolutional")

    # Create node config
    node_config = NodeConfig(input_dim=6, output_dim=1)

    # Create convolutional layer
    conv_layer = conv_layer_type(
        convolution_config=conv_config,
        node_type=node_type,
        node_kwargs=node_config,
        layer_type=layer_type,
        n_inputs_per_node=6,
        layer_config=layer_cfg,
    )

    # Create feedforward layer
    feedforward_layer = layer_type(
        input_size=16 * 14 * 14,
        output_size=50,
        node_type=node_type,
        node_kwargs=node_config,
        seed=43,
    )

    # Create groupsum
    groupsum_layer = GroupSum(k=2, tau=10.0, use_randperm=False)

    # Build model
    model = SimpleConvModel(conv_layer, feedforward_layer, groupsum_layer).to(device)

    # Train
    train_acc = train_model(
        model, train_images, train_labels, num_epochs=20, lr=0.01, device=device
    )

    # Test
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_images)
        _, test_predicted = test_outputs.max(1)
        test_accuracy = (test_predicted == test_labels).float().mean()

    # Success criterion: > 70% (well above random 50%)
    assert test_accuracy > 0.70, (
        f"Test accuracy {test_accuracy.item():.2%} is below threshold 70%. "
        f"Train accuracy: {train_acc:.2%}"
    )
