#!/usr/bin/env python3
"""
Fast model training test for DiffLUT.
Tests that a complete pipeline can train and reduce loss on a simple task.
- Minimal data (100 samples)
- Minimal epochs (3)
- Single configuration
- Measures: loss reduction, accuracy improvement

This test is designed for CI/CD pipelines:
- No verbose output beyond test status
- Deterministic results with fixed seeds
- Clear pass/fail criteria
- Exits with appropriate exit codes (0=pass, 1=fail)
"""

import sys
import warnings
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Suppress warnings for CI/CD - only show failures
warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut')
warnings.filterwarnings('ignore', category=UserWarning, module='difflut')

sys.path.insert(0, str(Path(__file__).parent.parent))

from difflut.registry import REGISTRY
from difflut.utils.modules import GroupSum
from difflut.nodes.node_config import NodeConfig


def generate_simple_data(n_samples=100, seed=42):
    """Generate simple binary classification data: XOR-like function."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    X = np.random.rand(n_samples, 8)
    # Simple rule: if sum of first 4 features > 2, class 1, else class 0
    y = (X[:, :4].sum(axis=1) > 2).astype(int)
    return X, y


class SimpleModel(nn.Module):
    """Minimal model: Input -> Encoder -> Layer -> GroupSum -> Output"""
    
    def __init__(self, encoder, layer, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.layer = layer
        self.groupsum = GroupSum(k=num_classes, tau=1.0, use_randperm=False)
    
    def forward(self, x):
        # Encode
        x_encoded = self.encoder.encode(x)
        if x_encoded.dim() == 3:
            x_encoded = x_encoded.reshape(x_encoded.shape[0], -1)
        x_encoded = torch.clamp(x_encoded, 0, 1)
        
        # Layer
        x_nodes = self.layer(x_encoded)
        
        # GroupSum
        out = self.groupsum(x_nodes)
        return out


def train_step(model, X, y, optimizer, criterion, device):
    """Single training step."""
    model.train()
    X_batch = X.to(device)
    y_batch = y.to(device)
    
    optimizer.zero_grad()
    logits = model(X_batch)
    loss = criterion(logits, y_batch)
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(model, X, y, device):
    """Evaluate accuracy."""
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        preds = torch.argmax(logits, dim=1).cpu()
        accuracy = (preds == y).float().mean().item()
    return accuracy


def test_model_training():
    """Test complete model training pipeline."""
    device = torch.device('cpu')
    
    try:
        # Generate data
        X_np, y_np = generate_simple_data(n_samples=100, seed=42)
        X_train = torch.tensor(X_np, dtype=torch.float32)
        y_train = torch.tensor(y_np, dtype=torch.long)
        
        # Create encoder
        encoder_class = REGISTRY.get_encoder('thermometer')
        encoder = encoder_class(num_bits=4, flatten=True)
        encoder.fit(X_train[:50])
        encoded_dim = encoder.encode(X_train[:1]).shape[1]
        
        # Create layer
        layer_class = REGISTRY.get_layer('random')
        node_class = REGISTRY.get_node('linear_lut')
        node_config = NodeConfig(input_dim=4, output_dim=1)
        
        layer = layer_class(
            input_size=encoded_dim,
            output_size=32,
            node_type=node_class,
            node_kwargs=node_config
        )
        
        # Create model
        model = SimpleModel(encoder=encoder, layer=layer, num_classes=2)
        model.to(device)
        
        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        accuracies = []
        
        for epoch in range(3):
            loss = train_step(model, X_train, y_train, optimizer, criterion, device)
            acc = evaluate(model, X_train, y_train, device)
            losses.append(loss)
            accuracies.append(acc)
        
        # Verify training worked
        initial_loss = losses[0]
        final_loss = losses[-1]
        loss_improvement = initial_loss - final_loss
        
        initial_acc = accuracies[0]
        final_acc = accuracies[-1]
        
        # Check success criteria
        success = True
        failures = []
        
        if loss_improvement <= 0:
            success = False
            failures.append("Loss did not decrease")
        
        if final_acc < 0.5:
            success = False
            failures.append("Final accuracy below 50%")
        
        if success:
            return True
        else:
            return False, failures
            
    except Exception as e:
        return False, [f"Exception: {str(e)}"]
        
        # Check success criteria
        success = True
        failures = []
        
        if loss_improvement <= 0:
            success = False
            failures.append("Loss did not decrease")
        
        if final_acc < 0.5:
            success = False
            failures.append("Final accuracy below 50%")
        
        if success:
            return True
        else:
            return False, failures
            
    except Exception as e:
        return False, [f"Exception: {str(e)}"]


def main():
    """Run test with CI/CD-friendly output."""
    try:
        # Run test
        result = test_model_training()
        
        # Handle result format
        if isinstance(result, tuple):
            success, failures = result
        else:
            success = result
            failures = []
        
        # Output for CI/CD
        if success:
            print("✓ PASS")
            sys.exit(0)
        else:
            print("✗ FAIL")
            for failure in failures:
                print(f"  {failure}")
            sys.exit(1)
    except Exception as e:
        print("✗ FAIL")
        print(f"  {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
