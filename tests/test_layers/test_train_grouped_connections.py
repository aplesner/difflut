"""
Test for grouped connections in ConvolutionalLayer.

The grouped connections feature allows LUTs to see multiple inputs from the same
channel within a receptive field, rather than randomly mixing across channels.

This test demonstrates the value by creating a task that is:
- Easy WITH grouped connections (pattern in single channel)
- Hard WITHOUT grouped connections (pattern lost in noise from other channels)
"""

import warnings

# Ignore general runtime warnings in tests
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pytest
import torch
import torch.nn as nn
from testing_utils import is_cuda_available

from difflut.utils.warnings import CUDAWarning, DefaultValueWarning

# Suppress DefaultValueWarnings and CUDAWarnings
warnings.filterwarnings("ignore", category=DefaultValueWarning)
warnings.filterwarnings("ignore", category=CUDAWarning)


IN_CHANNELS = 128
OUT_CHANNELS = 32
IMAGE_SIZE = 8
TRAIN_SAMPLES_PER_CLASS = 100
TEST_SAMPLES_PER_CLASS = 10
EPOCHS = 20


class SimpleConvModel(nn.Module):
    """Simple model: ConvolutionalLayer → RandomLayer → GroupSum."""

    def __init__(self, conv_layer, feedforward_layer, groupsum_layer):
        super().__init__()
        self.conv = conv_layer
        self.feedforward = feedforward_layer
        self.groupsum = groupsum_layer

    def forward(self, x):
        # Conv layer: (batch, channels, H, W) → (batch, out_channels, H', W')
        x = self.conv(x)

        # Flatten: (batch, out_channels, H', W') → (batch, out_channels*H'*W')
        x = x.reshape(x.shape[0], -1)

        # Random layer: (batch, flattened) → (batch, 50)
        x = self.feedforward(x)

        # GroupSum: (batch, 50) → (batch, 2)
        x = self.groupsum(x)

        return x


def create_single_channel_pattern_dataset(
    num_samples, image_size=IMAGE_SIZE, num_channels=IN_CHANNELS, pattern_channel=0
):
    """
    Create dataset where only one channel contains the pattern, rest is noise.

    This makes the task solvable ONLY if the model can compare multiple pixels
    within the same channel (enabled by grouped connections).

    Class 0: Vertical edges in pattern_channel (left=1, right=0)
    Class 1: Horizontal edges in pattern_channel (top=1, bottom=0)
    Other channels: Random noise (0 or 1)

    Args:
        num_samples: Number of samples per class
        image_size: Size of square images
        num_channels: Total number of channels
        pattern_channel: Which channel contains the pattern (default: 0)

    Returns:
        images: (2*num_samples, num_channels, image_size, image_size) float tensor
        labels: (2*num_samples,) long tensor
    """
    total_samples = 2 * num_samples

    # Initialize with zeros in all channels
    images = torch.zeros((total_samples, num_channels, image_size, image_size)).float()

    # Class 0: Vertical edges in pattern channel (num_samples images)
    images[:num_samples, pattern_channel, :, : image_size // 2] = 1.0  # Left half
    images[:num_samples, pattern_channel, :, image_size // 2 :] = 0.0  # Right half

    # Class 1: Horizontal edges in pattern channel (num_samples images)
    images[num_samples:, pattern_channel, : image_size // 2, :] = 1.0  # Top half
    images[num_samples:, pattern_channel, image_size // 2 :, :] = 0.0  # Bottom half

    # Create labels
    labels_class0 = torch.zeros(num_samples, dtype=torch.long)
    labels_class1 = torch.ones(num_samples, dtype=torch.long)
    labels = torch.cat([labels_class0, labels_class1], dim=0)

    # Shuffle
    perm = torch.randperm(total_samples)
    images = images[perm]
    labels = labels[perm]

    return images, labels


def train_model(model, train_images, train_labels, num_epochs=20, lr=0.01, device="cuda"):
    """Train model and return final training accuracy."""
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
# Test Fixtures
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
    train_images, train_labels = create_single_channel_pattern_dataset(
        num_samples=TRAIN_SAMPLES_PER_CLASS,
        image_size=IMAGE_SIZE,
        num_channels=IN_CHANNELS,
        pattern_channel=0,
    )
    test_images, test_labels = create_single_channel_pattern_dataset(
        num_samples=TEST_SAMPLES_PER_CLASS,
        image_size=IMAGE_SIZE,
        num_channels=IN_CHANNELS,
        pattern_channel=0,
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


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.training
@pytest.mark.parametrize(
    "use_grouped_connections,expected_min_accuracy,expected_max_accuracy",
    [
        (True, 0.85, 1.0),  # WITH grouped connections: should learn easily
        (False, 0.0, 0.60),  # WITHOUT grouped connections: should struggle
    ],
)
def test_grouped_connections_learning(
    use_grouped_connections,
    expected_min_accuracy,
    expected_max_accuracy,
    device,
    train_test_data,
):
    """
    Test that grouped connections enable learning from single-channel patterns.

    WITH grouped connections (GroupedInputConfig):
        - LUTs see multiple inputs from same channel
        - Can detect patterns within channel 0
        - Expected: >85% accuracy

    WITHOUT grouped connections:
        - LUTs see random mix of channels (mostly noise)
        - Cannot reliably detect pattern
        - Expected: <60% accuracy
    """
    from difflut import REGISTRY
    from difflut.layers import ConvolutionConfig, LayerConfig
    from difflut.nodes.node_config import NodeConfig
    from difflut.utils.modules import GroupSum

    train_images, train_labels, test_images, test_labels = train_test_data

    # Create layer config
    layer_cfg = LayerConfig()

    in_channels = IN_CHANNELS
    out_channels = OUT_CHANNELS

    # Create conv config with large chunk_size to disable chunking
    # (grouped connections may not work well with chunked first layer)
    conv_config = ConvolutionConfig(
        tree_depth=1,
        in_channels=in_channels,
        out_channels=out_channels,
        receptive_field=3,
        stride=1,
        padding=0,
        chunk_size=128,  # Larger than out_channels to disable chunking
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
        grouped_connections=use_grouped_connections,
        ensure_full_coverage=True,
    )

    # Calculate output size after convolution
    # Input: 8x8, Receptive field: 3x3, Stride: 1, Padding: 0
    # Output: (8 - 3) / 1 + 1 = 6x6
    conv_output_size = (
        out_channels * (IMAGE_SIZE - 2) * (IMAGE_SIZE - 2)
    )  # 32 channels, 6x6 spatial

    # Create feedforward layer
    feedforward_layer = layer_type(
        input_size=conv_output_size,
        output_size=50,
        node_type=node_type,
        node_kwargs=node_config,
        seed=43,
    )

    # Create groupsum
    groupsum_layer = GroupSum(k=2, tau=2.0, use_randperm=False)

    # Build model
    model = SimpleConvModel(conv_layer, feedforward_layer, groupsum_layer).to(device)

    # Train
    train_acc = train_model(
        model, train_images, train_labels, num_epochs=EPOCHS, lr=0.02, device=device
    )

    # Test
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_images)
        _, test_predicted = test_outputs.max(1)
        test_accuracy = (test_predicted == test_labels).float().mean().item()

    # Verify accuracy is in expected range
    config_name = (
        "WITH grouped connections" if use_grouped_connections else "WITHOUT grouped connections"
    )

    assert test_accuracy >= expected_min_accuracy, (
        f"{config_name}: Test accuracy {test_accuracy:.2%} is below minimum threshold {expected_min_accuracy:.0%}. "
        f"Train accuracy: {train_acc:.2%}"
    )

    assert test_accuracy <= expected_max_accuracy, (
        f"{config_name}: Test accuracy {test_accuracy:.2%} exceeds maximum threshold {expected_max_accuracy:.0%}. "
        f"This suggests the model learned unexpectedly well. Train accuracy: {train_acc:.2%}"
    )
