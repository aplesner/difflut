"""
Tests for model registry.

This file only tests that models are registered correctly.
Specific model tests are in their own files (test_feedforward.py, test_convolutional.py, etc.)
"""

import pytest

from difflut.registry import REGISTRY


def test_models_are_registered():
    """Test that at least some models are registered."""
    models = REGISTRY.list_models()
    assert len(models) > 0, "No models registered in REGISTRY"


def test_registry_get_model_works():
    """Test that get_model() works for all listed models."""
    for model_name in REGISTRY.list_models():
        model_class = REGISTRY.get_model(model_name)
        assert model_class is not None, f"get_model('{model_name}') returned None"


def test_registry_invalid_model_raises_error():
    """Test that getting invalid model raises ValueError."""
    with pytest.raises(ValueError):
        REGISTRY.get_model("nonexistent_model_12345")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
