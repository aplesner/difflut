#!/usr/bin/env python3
"""
Comprehensive test script for all registered layers and nodes in DiffLUT.
Tests classification of 10 highly distinctive function types.

Architecture:
- Input: 128 real values
- Encoder: distributive_thermometer (4 bits) -> 512 bits
- 1 Layer with 1000 nodes (tested in all combinations)
- GroupSum layer: 1000 -> 10 classes

Tests all combinations of:
- Registered layers (learnable, random, etc.)
- Registered nodes (dwn, fourier, linear_lut, etc.)

Generates:
1. Individual loss plots (layers x nodes grid)
2. Combined loss plot (all combinations)
3. Sample predictions visualization with bar plots
4. Text log of all results
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from difflut.registry import REGISTRY
from difflut.utils.modules import GroupSum

# ==================== Constants (Configurable) ====================
NUM_NODES = 200            # Number of nodes in the layer
NUM_NODE_INPUTS = 6         # Number of inputs per node (will select randomly from encoded_dim)
LEARNING_RATE = 0.1        # Learning rate for Adam optimizer
EPOCHS = 2                 # Training epochs
BATCH_SIZE = 16             # Batch size for training
THERMOMETER_BITS = 3        # Number of bits for thermometer encoding
N_TRAIN = 800               # Number of training samples
RANDOM_SEED = 42            # Random seed for reproducibility
# Note: NUM_CLASSES = 10 is fixed by the 10 function types in generate_highly_distinct_functions()


class TeeLogger:
    """Logger that writes to both stdout and a file."""
    def __init__(self, log_file):
        self.log_file = open(log_file, 'w')
        self.stdout = sys.stdout
    
    def write(self, message):
        self.stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.stdout.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


# ==================== Function Generation ====================

def generate_highly_distinct_functions(num_samples=1000, sequence_length=128, seed=42):
    """Generate highly distinctive functions for 10-class classification."""
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
            steepness = np.random.uniform(2.0, 3.0)
            y_vals = np.sqrt(steepness * x_shifted)
            y_vals = (y_vals - np.min(y_vals)) / (np.max(y_vals) - np.min(y_vals))
            
        elif func_type == 1: # linear_up
            slope = np.random.uniform(0.08, 0.12)
            intercept = np.random.uniform(-1.0, 0.0)
            y_vals = slope * x + intercept
            y_vals = (y_vals - np.min(y_vals)) / (np.max(y_vals) - np.min(y_vals))
            
        elif func_type == 2: # linear_flat
            slope = np.random.uniform(-0.001, 0.001)
            intercept = np.random.uniform(0.4, 0.6)
            y_vals = slope * x + intercept
            y_vals = np.clip(y_vals, 0, 1)
            
        elif func_type == 3: # linear_down
            slope = np.random.uniform(-0.12, -0.08)
            intercept = np.random.uniform(1.0, 2.0)
            y_vals = slope * x + intercept
            y_vals = (y_vals - np.min(y_vals)) / (np.max(y_vals) - np.min(y_vals))
            
        elif func_type == 4: # quadratic
            a = np.random.uniform(0.01, 0.02)
            vertex_x = np.random.uniform(-3, 3)
            vertex_y = np.random.uniform(0.0, 0.2)
            y_vals = a * (x - vertex_x)**2 + vertex_y
            y_vals = (y_vals - np.min(y_vals)) / (np.max(y_vals) - np.min(y_vals))
            
        elif func_type == 5: # absolute
            vertex_x = np.random.uniform(-2, 2)
            slope = np.random.uniform(0.08, 0.12)
            y_vals = slope * np.abs(x - vertex_x)
            y_vals = (y_vals - np.min(y_vals)) / (np.max(y_vals) - np.min(y_vals))
            
        elif func_type == 6: # step
            step_position = np.random.uniform(-5, 5)
            low_value = np.random.uniform(0.0, 0.2)
            high_value = np.random.uniform(0.8, 1.0)
            y_vals = np.where(x < step_position, low_value, high_value)
            
        elif func_type == 7: # triangle
            period = np.random.uniform(4, 5)
            amplitude = np.random.uniform(0.4, 0.5)
            center = np.random.uniform(0.4, 0.6)
            triangle_wave = 2 * amplitude / period * np.abs((x % period) - period/2) - amplitude + center
            y_vals = np.clip(triangle_wave, 0, 1)
            
        elif func_type == 8: # sinus
            frequency = np.random.uniform(0.5, 0.8)
            amplitude = np.random.uniform(0.4, 0.45)
            phase = np.random.uniform(0, 2*np.pi)
            y_vals = amplitude * np.sin(frequency * x + phase) + 0.5
            y_vals = np.clip(y_vals, 0, 1)
            
        elif func_type == 9: # cosinus
            frequency = np.random.uniform(0.5, 0.8)
            amplitude = np.random.uniform(0.4, 0.45)
            phase = np.random.uniform(0, 2*np.pi)
            y_vals = amplitude * np.cos(frequency * x + phase) + 0.5
            y_vals = np.clip(y_vals, 0, 1)
        
        # Add noise and quantize to 4-bit
        noise = np.random.normal(0, 0.005, sequence_length)
        y_vals = np.clip(y_vals + noise, 0, 1)
        y_vals_4bit = np.round(y_vals * 15) / 15
        
        X.append(y_vals_4bit)
        y.append(func_type)
    
    return np.array(X), np.array(y), function_names


# ==================== Model Definition ====================

class DiffLUTClassificationModel(nn.Module):
    """
    Simple classification model for function types.
    
    Architecture:
    - Input (128) -> Thermometer Encoder -> encoded_dim
    - 1 Layer with nodes
    - GroupSum layer -> 10 classes
    """
    
    def __init__(self, encoder, layer, num_nodes=NUM_NODES, num_classes=10):
        """
        Args:
            encoder: Encoder instance (distributive_thermometer)
            layer: Layer instance (already constructed with output_size=num_nodes)
            num_nodes: Number of nodes in the layer
            num_classes: Number of output classes
        """
        super().__init__()
        self.encoder = encoder
        self.layer = layer
        self.num_nodes = num_nodes
        self.groupsum = GroupSum(k=num_classes, tau=1)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, 128)
            
        Returns:
            Output logits (batch_size, 10)
        """
        batch_size = x.shape[0]
        
        # Flatten input if needed (for compatibility with different input shapes)
        if x.dim() > 2:
            x = x.view(batch_size, -1)
        
        # Move encoder to same device as input
        self.encoder.to(x.device)
        
        # Encode - should return 2D (batch_size, num_features * num_bits)
        x_encoded = self.encoder.encode(x)
        
        # Safety check: flatten if encoder somehow returns 3D
        if x_encoded.dim() == 3:
            x_encoded = x_encoded.reshape(batch_size, -1)
        
        # Clamp to valid range [0, 1]
        x_encoded = torch.clamp(x_encoded, 0, 1)
        
        # Layer with nodes
        x_nodes = self.layer(x_encoded)  # (batch_size, num_nodes)
        
        # GroupSum: (batch_size, num_nodes) -> (batch_size, 10)
        out = self.groupsum(x_nodes)  # (batch_size, 10)
        
        return out


# ==================== Training ====================

def train_model(
    model: nn.Module,
    train_loader,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    device: torch.device = torch.device('cpu')
) -> List[float]:
    """
    Train a model and track loss history.
    
    Returns:
        List of training losses per epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    train_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        train_loss /= num_batches
        train_losses.append(train_loss)
    
    return train_losses


def get_predictions(
    model: nn.Module,
    X: torch.Tensor,
    device: torch.device = torch.device('cpu')
) -> np.ndarray:
    """Get predictions on data."""
    model.eval()
    
    with torch.no_grad():
        X_device = X.to(device)
        logits = model(X_device)
        probs = torch.softmax(logits, dim=1)
    
    return probs.cpu().numpy()


# ==================== Visualization ====================

def plot_loss_curves_grid(
    results_dict: Dict,
    layer_names: List[str],
    node_names: List[str],
    save_path: Path
):
    """
    Plot individual loss curves in a grid (layers x nodes).
    """
    n_layers = len(layer_names)
    n_nodes = len(node_names)
    
    fig, axes = plt.subplots(n_layers, n_nodes, figsize=(3*n_nodes, 2.5*n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    if n_nodes == 1:
        axes = axes.reshape(-1, 1)
    
    for layer_idx, layer_name in enumerate(layer_names):
        for node_idx, node_name in enumerate(node_names):
            ax = axes[layer_idx, node_idx]
            key = (layer_name, node_name)
            
            if key in results_dict:
                result = results_dict[key]
                losses = result['train_losses']
                ax.plot(losses, linewidth=2, color='steelblue')
                ax.set_ylabel('Loss', fontsize=8)
                ax.set_xlabel('Epoch', fontsize=8)
                ax.set_title(f'{layer_name}\n×\n{node_name}', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                ax.tick_params(labelsize=7)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
                ax.set_title(f'{layer_name}\n×\n{node_name}', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Individual loss curves saved: {save_path}")


def plot_loss_curves_combined(
    results_dict: Dict,
    layer_names: List[str],
    node_names: List[str],
    save_path: Path
):
    """
    Plot all loss curves in a single figure for comparison.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(results_dict)))
    
    for idx, (key, result) in enumerate(results_dict.items()):
        layer_name, node_name = key
        losses = result['train_losses']
        label = f'{layer_name} × {node_name}'
        ax.plot(losses, label=label, color=colors[idx], alpha=0.8, linewidth=1.5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss - All Layer-Node Combinations', fontsize=13, weight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best', ncol=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Combined loss curves saved: {save_path}")


def plot_predictions_grid(
    X_samples: np.ndarray,
    y_true: np.ndarray,
    predictions_dict: Dict,
    layer_names: List[str],
    node_names: List[str],
    function_names: List[str],
    save_path: Path = None
):
    """
    Plot 10 sample inputs (one per class) with prediction bar plots.
    
    Layout: For each class, show sample and prediction bars for each combination.
    """
    n_classes = len(function_names)
    n_combos = len(predictions_dict)
    
    # Get one sample per class
    samples_per_class = []
    for class_idx in range(n_classes):
        indices = np.where(y_true == class_idx)[0]
        if len(indices) > 0:
            samples_per_class.append(indices[0])
    
    n_rows = len(samples_per_class)
    n_cols = 1 + n_combos  # 1 for the input, rest for predictions
    
    fig = plt.figure(figsize=(2.5*n_cols, 2.5*n_rows))
    
    for row_idx, sample_idx in enumerate(samples_per_class):
        true_label = y_true[sample_idx]
        x_sample = X_samples[sample_idx]
        
        # Plot input function
        ax_input = plt.subplot(n_rows, n_cols, row_idx * n_cols + 1)
        x_axis = np.linspace(-10, 10, len(x_sample))
        ax_input.plot(x_axis, x_sample, 'b-', linewidth=2)
        ax_input.set_title(f'{function_names[true_label]}', fontsize=10, weight='bold')
        ax_input.set_ylim(-0.05, 1.05)
        ax_input.set_ylabel('Value', fontsize=8)
        ax_input.grid(True, alpha=0.3)
        ax_input.set_xticks([])
        
        # Plot predictions for each combination
        col_offset = 1
        for (layer_name, node_name), probs in predictions_dict.items():
            ax_pred = plt.subplot(n_rows, n_cols, row_idx * n_cols + col_offset + 1)
            
            pred_probs = probs[sample_idx]
            colors_bar = ['green' if i == true_label else 'lightblue' for i in range(n_classes)]
            
            ax_pred.bar(range(n_classes), pred_probs, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax_pred.set_ylim(0, 1)
            ax_pred.set_xticks(range(n_classes))
            ax_pred.set_xticklabels([i for i in range(n_classes)], fontsize=6)
            ax_pred.set_title(f'{layer_name}\n×\n{node_name}', fontsize=8)
            ax_pred.set_yticks([0, 0.5, 1])
            ax_pred.tick_params(labelsize=6)
            
            col_offset += 1
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Prediction visualization saved: {save_path}")


# ==================== Main Test Function ====================

def main():
    """Main test function."""
    # Setup output directory
    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"test_layers_nodes_{timestamp}.log"
    logger = TeeLogger(str(log_file))
    sys.stdout = logger
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("="*70)
        print("DiffLUT LAYERS & NODES CLASSIFICATION TEST - SIMPLIFIED")
        print("="*70)
        print(f"\nDevice: {device}")
        print(f"Output directory: {output_dir}\n")
        
        print(f"Configuration:")
        print(f"  - Epochs: {EPOCHS}")
        print(f"  - Batch size: {BATCH_SIZE}")
        print(f"  - Learning rate: {LEARNING_RATE}")
        print(f"  - Train samples: {N_TRAIN}")
        print(f"  - Thermometer bits: {THERMOMETER_BITS}")
        print(f"  - Node input dim: {NUM_NODE_INPUTS}")
        print(f"  - Nodes per layer: {NUM_NODES}")
        print(f"  - Output classes: 10 (fixed by function types)\n")
        
        # Generate data
        print("Generating training data...")
        X_train, y_train, function_names = generate_highly_distinct_functions(N_TRAIN, seed=RANDOM_SEED)
        NUM_CLASSES = len(function_names)  # 10 classes
        print(f"✓ Generated {len(X_train)} samples")
        print(f"  Function types: {function_names}\n")
        
        # Create data loader
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Get available layers and nodes
        layer_names = sorted(REGISTRY.list_layers())
        node_names = sorted(REGISTRY.list_nodes())
        
        print(f"Available layers ({len(layer_names)}): {layer_names}")
        print(f"Available nodes ({len(node_names)}): {node_names}\n")
        
        # Get encoder
        print("Creating thermometer encoder...")
        encoder_class = REGISTRY.get_encoder('distributive_thermometer')
        X_sample_torch = torch.tensor(X_train[:100], dtype=torch.float32)
        encoder = encoder_class(num_bits=THERMOMETER_BITS)
        encoder.fit(X_sample_torch)
        encoded_dim = encoder.encode(X_sample_torch[:1]).shape[1]
        print(f"✓ Encoder created (output dim: {encoded_dim})\n")
        
        # Test all layer-node combinations
        print("Training all layer-node combinations...")
        print("="*70)
        
        results = {}
        predictions = {}
        
        for layer_idx, layer_name in enumerate(layer_names):
            for node_idx, node_name in enumerate(node_names):
                key = (layer_name, node_name)
                combo_num = layer_idx * len(node_names) + node_idx + 1
                total_combos = len(layer_names) * len(node_names)
                
                print(f"\n[{combo_num}/{total_combos}] {layer_name:12s} × {node_name:15s}", end=" ")
                sys.stdout.flush()
                
                try:
                    # Get classes
                    layer_class = REGISTRY.get_layer(layer_name)
                    node_class = REGISTRY.get_node(node_name)
                    
                    # Create layer with nodes
                    # Note: node_kwargs should have input_dim as integer (not list)
                    layer = layer_class(
                        input_size=encoded_dim,
                        output_size=NUM_NODES,
                        node_type=node_class,
                        node_kwargs={'input_dim': NUM_NODE_INPUTS, 'output_dim': 1}
                    )
                    
                    # Create model
                    model = DiffLUTClassificationModel(
                        encoder=encoder,
                        layer=layer,
                        num_classes=10
                    )
                    
                    # Train
                    train_losses = train_model(
                        model, train_loader,
                        epochs=EPOCHS, lr=LEARNING_RATE, device=device
                    )
                    
                    # Store results
                    results[key] = {
                        'train_losses': train_losses,
                        'final_loss': train_losses[-1]
                    }
                    
                    # Get predictions on training set (10 samples)
                    X_sample = torch.tensor(X_train[:10], dtype=torch.float32)
                    probs = get_predictions(model, X_sample, device)
                    predictions[key] = probs
                    
                    print(f"✓ Loss: {train_losses[-1]:.6f}")
                    
                except Exception as e:
                    print(f"✗ Error: {str(e)[:50]}")
                    results[key] = {'error': str(e)}
        
        # Generate visualizations
        print("\n" + "="*70)
        print("Generating visualizations...")
        print("="*70)
        
        # Filter to successful results
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        successful_predictions = {k: v for k, v in predictions.items() if k in successful_results}
        successful_layer_names = sorted(set(k[0] for k in successful_results.keys()))
        successful_node_names = sorted(set(k[1] for k in successful_results.keys()))
        
        if successful_results:
            # 1. Individual loss curves
            print("\nGenerating individual loss curves grid...")
            loss_grid_path = output_dir / f"loss_curves_grid_{timestamp}.png"
            plot_loss_curves_grid(successful_results, successful_layer_names, successful_node_names, loss_grid_path)
            
            # 2. Combined loss curves
            print("Generating combined loss comparison...")
            loss_combined_path = output_dir / f"loss_curves_combined_{timestamp}.png"
            plot_loss_curves_combined(successful_results, successful_layer_names, successful_node_names, loss_combined_path)
            
            # 3. Prediction visualization (first 10 samples from training set)
            print("Generating prediction visualization...")
            pred_path = output_dir / f"predictions_grid_{timestamp}.png"
            X_sample_10 = X_train[:10]
            y_sample_10 = y_train[:10]
            plot_predictions_grid(
                X_sample_10, y_sample_10, successful_predictions,
                successful_layer_names, successful_node_names,
                function_names,
                save_path=pred_path
            )
        
        # Print final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        
        successful = [r for r in results.values() if 'error' not in r]
        failed = [r for r in results.values() if 'error' in r]
        
        print(f"\n✓ Successful: {len(successful)}/{len(results)} combinations")
        print(f"✗ Failed: {len(failed)}/{len(results)} combinations")
        
        if successful:
            print("\nBest performers (lowest final training loss):")
            sorted_results = sorted(
                [(k, v) for k, v in results.items() if 'error' not in v],
                key=lambda x: x[1]['final_loss']
            )
            for idx, (key, result) in enumerate(sorted_results[:5], 1):
                layer_name, node_name = key
                print(f"  {idx}. {layer_name:12s} × {node_name:15s} "
                      f"(Loss: {result['final_loss']:.6f})")
        
        print("\n" + "="*70)
        print(f"All outputs saved to: {output_dir}")
        print(f"Log file: {log_file}")
        print("="*70)
        
        return 0 if len(failed) == 0 else 1
        
    finally:
        logger.close()
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
