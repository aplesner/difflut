#!/usr/bin/env python3
"""
Simplified test script for DiffLUT with specific configuration:
- Encoder: distributive_thermometer
- Node: probabilistic
- Layer: random

Tests basic classification on 10 function types with minimal configuration.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from difflut.registry import REGISTRY
from difflut.utils.modules import GroupSum
from difflut.nodes.node_config import NodeConfig

# ==================== Configuration ====================
NUM_NODES = 500
NUM_NODE_INPUTS = 8
LEARNING_RATE = 0.05
EPOCHS = 5
BATCH_SIZE = 32
THERMOMETER_BITS = 4
N_TRAIN = 500
RANDOM_SEED = 42
NUM_CLASSES = 10


# ==================== Data Generation ====================

def generate_simple_functions(num_samples=500, sequence_length=128, seed=42):
    """Generate 10 distinctive function types."""
    np.random.seed(seed)
    X = []
    y = []
    
    function_names = [
        'sqrt', 'linear_up', 'linear_flat', 'linear_down', 'quadratic',
        'absolute', 'step', 'triangle', 'sinus', 'cosinus'
    ]
    
    x = np.linspace(-10, 10, sequence_length)
    
    for i in range(num_samples):
        func_type = np.random.choice(len(function_names))
        
        if func_type == 0:   # sqrt
            x_shifted = x - np.min(x) + 0.1
            y_vals = np.sqrt(2.5 * x_shifted)
            
        elif func_type == 1: # linear_up
            y_vals = 0.1 * x
            
        elif func_type == 2: # linear_flat
            y_vals = 0.5 * np.ones_like(x)
            
        elif func_type == 3: # linear_down
            y_vals = -0.1 * x + 1.0
            
        elif func_type == 4: # quadratic
            y_vals = 0.015 * x**2
            
        elif func_type == 5: # absolute
            y_vals = 0.1 * np.abs(x)
            
        elif func_type == 6: # step
            y_vals = np.where(x < 0, 0.2, 0.8)
            
        elif func_type == 7: # triangle
            y_vals = 0.4 * (1 - np.abs(x % 4 - 2) / 2) + 0.3
            
        elif func_type == 8: # sinus
            y_vals = 0.4 * np.sin(0.6 * x) + 0.5
            
        elif func_type == 9: # cosinus
            y_vals = 0.4 * np.cos(0.6 * x) + 0.5
        
        # Normalize and add noise
        y_vals = (y_vals - np.min(y_vals)) / (np.max(y_vals) - np.min(y_vals) + 1e-8)
        noise = np.random.normal(0, 0.01, sequence_length)
        y_vals = np.clip(y_vals + noise, 0, 1)
        
        X.append(y_vals)
        y.append(func_type)
    
    return np.array(X), np.array(y), function_names


# ==================== Model ====================

class SimpleClassificationModel(nn.Module):
    """Simple model: Input -> Encoder -> Layer -> GroupSum -> Classes"""
    
    def __init__(self, encoder, layer, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.layer = layer
        self.groupsum = GroupSum(k=num_classes, tau=1)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(batch_size, -1)
        
        # Encode
        self.encoder.to(x.device)
        x_encoded = self.encoder.encode(x)
        
        # Flatten if 3D
        if x_encoded.dim() == 3:
            x_encoded = x_encoded.reshape(batch_size, -1)
        
        x_encoded = torch.clamp(x_encoded, 0, 1)
        
        # Layer
        x_nodes = self.layer(x_encoded)
        
        # GroupSum
        out = self.groupsum(x_nodes)
        
        return out


# ==================== Training ====================

def train_model(model, train_loader, epochs, lr, device):
    """Train and return loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
    
    return losses


def evaluate_model(model, X, y, device):
    """Evaluate model accuracy."""
    model.eval()
    with torch.no_grad():
        X_device = X.to(device)
        logits = model(X_device)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    
    accuracy = np.mean(preds == y.numpy())
    return accuracy


# ==================== Visualization ====================

def plot_results(losses, X_samples, y_true, model, function_names, device, save_path):
    """Plot training loss and sample predictions."""
    fig = plt.figure(figsize=(15, 8))
    
    # Plot 1: Training loss
    ax1 = plt.subplot(2, 5, 1)
    ax1.plot(losses, 'o-', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2-10: Sample predictions (one per class)
    model.eval()
    with torch.no_grad():
        X_samples_device = torch.tensor(X_samples, dtype=torch.float32).to(device)
        logits = model(X_samples_device)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    
    for i in range(9):
        ax = plt.subplot(2, 5, i+2)
        
        if i < len(X_samples):
            # Show input function
            x_axis = np.linspace(-10, 10, len(X_samples[i]))
            ax.plot(x_axis, X_samples[i], 'b-', linewidth=2, alpha=0.7)
            ax.set_ylim(-0.05, 1.05)
            
            true_label = y_true[i]
            pred_label = np.argmax(probs[i])
            
            title_color = 'green' if pred_label == true_label else 'red'
            ax.set_title(f'{function_names[true_label]}\nPred: {pred_label}', 
                        fontsize=9, color=title_color, weight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Results saved: {save_path}")


# ==================== Main ====================

def main():
    """Main test function."""
    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("SIMPLE TEST: distributive_thermometer + probabilistic + random")
    print("="*70)
    print(f"Device: {device}")
    print(f"Configuration:")
    print(f"  - Encoder: distributive_thermometer ({THERMOMETER_BITS} bits)")
    print(f"  - Node: probabilistic")
    print(f"  - Layer: random")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Samples: {N_TRAIN}")
    print(f"  - Nodes: {NUM_NODES}")
    print(f"  - Learning rate: {LEARNING_RATE}\n")
    
    # Generate data
    print("Generating data...")
    X_train, y_train, function_names = generate_simple_functions(N_TRAIN, seed=RANDOM_SEED)
    print(f"✓ Generated {len(X_train)} samples")
    print(f"  Classes: {function_names}\n")
    
    # Create data loader
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create encoder
    print("Creating encoder...")
    encoder_class = REGISTRY.get_encoder('distributive_thermometer')
    X_sample = torch.tensor(X_train[:100], dtype=torch.float32)
    encoder = encoder_class(num_bits=THERMOMETER_BITS)
    encoder.fit(X_sample)
    encoded_dim = encoder.encode(X_sample[:1]).shape[1]
    print(f"✓ Encoder created (output dim: {encoded_dim})\n")
    
    # Create layer with nodes
    print("Creating layer...")
    layer_class = REGISTRY.get_layer('random')
    node_class = REGISTRY.get_node('probabilistic')
    
    node_config = NodeConfig(
        input_dim=NUM_NODE_INPUTS,
        output_dim=1
    )
    
    layer = layer_class(
        input_size=encoded_dim,
        output_size=NUM_NODES,
        node_type=node_class,
        node_kwargs=node_config
    )
    print(f"✓ Layer created (random × probabilistic)\n")
    
    # Create model
    print("Creating model...")
    model = SimpleClassificationModel(encoder=encoder, layer=layer, num_classes=NUM_CLASSES)
    print(f"✓ Model created\n")
    
    # Train
    print("Training...")
    print("-"*70)
    losses = train_model(model, train_loader, EPOCHS, LEARNING_RATE, device)
    print("-"*70)
    print(f"✓ Training complete. Final loss: {losses[-1]:.6f}\n")
    
    # Evaluate
    print("Evaluating...")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    accuracy = evaluate_model(model, X_train_tensor, y_train_tensor, device)
    print(f"✓ Training accuracy: {accuracy:.2%}\n")
    
    # Visualize
    print("Generating visualization...")
    # Get one sample per class
    samples = []
    labels = []
    for class_idx in range(NUM_CLASSES):
        indices = np.where(y_train == class_idx)[0]
        if len(indices) > 0:
            samples.append(X_train[indices[0]])
            labels.append(y_train[indices[0]])
    
    save_path = output_dir / f"simple_test_{timestamp}.png"
    plot_results(losses, np.array(samples), np.array(labels), model, function_names, device, save_path)
    
    print("="*70)
    print("TEST COMPLETE")
    print("="*70)
    print(f"Final Loss: {losses[-1]:.6f}")
    print(f"Training Accuracy: {accuracy:.2%}")
    print(f"Output: {save_path}")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
