#!/usr/bin/env python3
"""
Example script demonstrating the new DiffLUT model system.

This script shows how to:
1. Create a model from config
2. Load a pretrained model
3. Use runtime overrides
4. Save and load models
"""

import sys
from pathlib import Path

import torch

# Add difflut to path (adjust as needed)
sys.path.insert(0, str(Path(__file__).parent.parent / "difflut"))

from difflut.models import (
    ModelConfig,
    SimpleFeedForward,
    build_model,
    get_pretrained_model_info,
    list_pretrained_models,
)


def example_1_create_from_config():
    """Example 1: Create a model from ModelConfig."""
    print("\n" + "=" * 60)
    print("Example 1: Create Model from Config")
    print("=" * 60)

    config = ModelConfig(
        model_type="feedforward",
        layer_type="random",
        node_type="probabilistic",
        encoder_config={"name": "thermometer", "num_bits": 4, "feature_wise": True},
        node_input_dim=6,
        layer_widths=[128, 64],
        num_classes=10,
        input_size=784,
        dataset="mnist",
        runtime={"temperature": 1.0, "eval_mode": "expectation"},
    )

    print(f"Config: {config}")

    model = SimpleFeedForward(config)
    print(f"\nModel: {model}")

    # Fit encoder on dummy data
    dummy_data = torch.randn(100, 784)
    model.fit_encoder(dummy_data)

    # Test forward pass
    test_input = torch.randn(10, 784)
    output = model(test_input)
    print(f"\nInput shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    params = model.count_parameters()
    print(f"\nParameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")


def example_2_list_pretrained():
    """Example 2: List available pretrained models."""
    print("\n" + "=" * 60)
    print("Example 2: List Pretrained Models")
    print("=" * 60)

    models = list_pretrained_models()

    if models:
        for model_type, names in models.items():
            print(f"\n{model_type}:")
            for name in names:
                print(f"  - {name}")
    else:
        print("No pretrained models found.")


def example_3_load_pretrained():
    """Example 3: Load a pretrained model (config only)."""
    print("\n" + "=" * 60)
    print("Example 3: Load Pretrained Model (Config Only)")
    print("=" * 60)

    try:
        # Try to load mnist_large (config exists, weights may not)
        model = build_model("mnist_large", load_weights=False)
        print(f"Loaded model: {model.__class__.__name__}")
        print(f"Config: {model.config}")

        # Get model info
        info = get_pretrained_model_info("mnist_large")
        print(f"\nModel Info:")
        print(f"  Model type: {info['model_type']}")
        print(f"  Layer widths: {info['layer_widths']}")
        print(f"  Num classes: {info['num_classes']}")
        print(f"  Dataset: {info['dataset']}")
        print(f"  Has weights: {info['has_weights']}")

    except FileNotFoundError as e:
        print(f"Could not load pretrained model: {e}")


def example_4_runtime_overrides():
    """Example 4: Use runtime overrides."""
    print("\n" + "=" * 60)
    print("Example 4: Runtime Overrides")
    print("=" * 60)

    config = ModelConfig(
        model_type="feedforward",
        layer_type="random",
        node_type="probabilistic",
        encoder_config={"name": "thermometer", "num_bits": 4},
        node_input_dim=6,
        layer_widths=[128],
        num_classes=10,
        input_size=784,
        runtime={"temperature": 1.0},
    )

    model = SimpleFeedForward(config)
    print(f"Initial temperature: {model.runtime.get('temperature')}")

    # Apply runtime overrides
    model.apply_runtime_overrides({"temperature": 0.5, "eval_mode": "sampling"})

    print(f"Updated temperature: {model.runtime.get('temperature')}")
    print(f"Updated eval_mode: {model.runtime.get('eval_mode')}")


def example_5_save_and_load():
    """Example 5: Save and load model."""
    print("\n" + "=" * 60)
    print("Example 5: Save and Load Model")
    print("=" * 60)

    # Create model
    config = ModelConfig(
        model_type="feedforward",
        layer_type="random",
        node_type="probabilistic",
        encoder_config={"name": "thermometer", "num_bits": 4},
        node_input_dim=6,
        layer_widths=[128],
        num_classes=10,
        input_size=784,
    )

    model = SimpleFeedForward(config)

    # Fit encoder
    dummy_data = torch.randn(100, 784)
    model.fit_encoder(dummy_data)

    # Save config
    config_path = "/tmp/test_model_config.yaml"
    model.save_config(config_path)
    print(f"Saved config to: {config_path}")

    # Save weights
    weights_path = "/tmp/test_model_weights.pth"
    model.save_weights(weights_path)
    print(f"Saved weights to: {weights_path}")

    # Load config
    loaded_config = ModelConfig.from_yaml(config_path)
    print(f"\nLoaded config: {loaded_config}")

    # Create new model and load weights
    new_model = SimpleFeedForward(loaded_config)
    new_model.fit_encoder(dummy_data)  # Need to fit encoder first
    new_model.load_weights(weights_path)
    print(f"Created new model and loaded weights")


def example_6_yaml_config():
    """Example 6: Build from YAML config."""
    print("\n" + "=" * 60)
    print("Example 6: Build from YAML Config")
    print("=" * 60)

    try:
        # Try to build from the mnist_large example config
        yaml_path = (
            Path(__file__).parent.parent
            / "difflut"
            / "difflut"
            / "models"
            / "pretrained"
            / "feedforward"
            / "mnist_large.yaml"
        )

        if yaml_path.exists():
            model = build_model(str(yaml_path), load_weights=False)
            print(f"Built model from YAML: {yaml_path.name}")
            print(f"Model: {model}")

            # Show some config details
            config = model.get_config()
            print(f"\nConfig details:")
            print(f"  Node type: {config.node_type}")
            print(f"  Layer widths: {config.layer_widths}")
            print(f"  Encoder: {config.encoder_config.get('name')}")
        else:
            print(f"Config file not found: {yaml_path}")

    except Exception as e:
        print(f"Error building from YAML: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("DiffLUT Model System Examples")
    print("=" * 60)

    try:
        example_1_create_from_config()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        example_2_list_pretrained()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        example_3_load_pretrained()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    try:
        example_4_runtime_overrides()
    except Exception as e:
        print(f"Example 4 failed: {e}")

    try:
        example_5_save_and_load()
    except Exception as e:
        print(f"Example 5 failed: {e}")

    try:
        example_6_yaml_config()
    except Exception as e:
        print(f"Example 6 failed: {e}")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
