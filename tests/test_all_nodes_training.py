#!/usr/bin/env python3
"""
Comprehensive test script for all registered nodes in DiffLUT.
Tests:
1. All nodes can be instantiated
2. Loss decreases during training for at least one learning rate
3. Generates 3D surface plots for XOR function approximation
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


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

from difflut.registry import REGISTRY


# Continuous XOR function: xorc(x, y) = x + y - 2xy
def xorc(x, y):
    """Continuous XOR function for testing."""
    return x + y - 2 * x * y


def generate_training_data(n_samples=1000, seed=42):
    """Generate random training data for XOR."""
    np.random.seed(seed)
    X_np = np.random.rand(n_samples, 2)
    Y_np = xorc(X_np[:, 0], X_np[:, 1])
    # Reshape to (batch_size, 1, input_dim) for 3D input
    X_train = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1)  # (n_samples, 1, 2)
    Y_train = torch.tensor(Y_np, dtype=torch.float32).unsqueeze(-1)  # (n_samples, 1)
    return X_train, Y_train


def create_node_instance(node_class):
    """
    Create a node instance with appropriate kwargs for each node type.
    All nodes now have sensible defaults, so we just pass input_dim and output_dim.
    
    Args:
        node_class: The node class to instantiate
        
    Returns:
        Instance of the node
    """
    # All nodes accept input_dim and output_dim with defaults for other parameters
    node_kwargs = dict(input_dim=2, output_dim=1)
    
    return node_class(**node_kwargs)


def train_node_with_lr(node, X_train, Y_train, lr, epochs=10, verbose=False):
    """
    Train a node with a specific learning rate.
    
    Args:
        node: Node instance to train
        X_train: Training inputs (batch_size, 1, input_dim)
        Y_train: Training targets (batch_size, 1)
        lr: Learning rate
        epochs: Number of training epochs
        verbose: Print progress
        
    Returns:
        Tuple of (initial_loss, final_loss, loss_history)
    """
    optimizer = torch.optim.Adam(node.parameters(), lr=lr)
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = node(X_train)
        
        # Flatten output to match target shape if needed
        y_pred = y_pred.reshape(Y_train.shape)
        
        loss = torch.nn.functional.mse_loss(y_pred, Y_train)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:3d}: Loss = {loss.item():.6f}")
    
    return loss_history[0], loss_history[-1], loss_history


def test_node(node_name, node_class, X_train, Y_train, learning_rates, epochs=100):
    """
    Test a node with multiple learning rates and find the best one.
    
    Args:
        node_name: Name of the node
        node_class: Node class
        X_train: Training inputs
        Y_train: Training targets
        learning_rates: List of learning rates to try
        epochs: Number of epochs per trial
        
    Returns:
        Dict with test results
    """
    print(f"\n{'='*60}")
    print(f"Testing: {node_name}")
    print(f"{'='*60}")
    
    results = {
        'node_name': node_name,
        'success': False,
        'best_lr': None,
        'initial_loss': None,
        'final_loss': None,
        'loss_decrease': None,
        'best_node': None,
        'loss_history': None,
        'all_results': []
    }
    
    best_decrease = -float('inf')
    
    for lr in learning_rates:
        print(f"\nTrying learning rate: {lr}")
        
        try:
            # Create fresh node instance
            node = create_node_instance(node_class)
            
            # Train the node
            initial_loss, final_loss, loss_history = train_node_with_lr(
                node, X_train, Y_train, lr, epochs, verbose=False
            )
            
            loss_decrease = initial_loss - final_loss
            improvement_pct = (loss_decrease / initial_loss) * 100 if initial_loss > 0 else 0
            
            print(f"  Initial loss: {initial_loss:.6f}")
            print(f"  Final loss:   {final_loss:.6f}")
            print(f"  Decrease:     {loss_decrease:.6f} ({improvement_pct:.2f}% improvement)")
            
            # Store results
            lr_result = {
                'lr': lr,
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'loss_decrease': loss_decrease,
                'improvement_pct': improvement_pct
            }
            results['all_results'].append(lr_result)
            
            # Update best if this is better
            if loss_decrease > best_decrease:
                best_decrease = loss_decrease
                results['best_lr'] = lr
                results['initial_loss'] = initial_loss
                results['final_loss'] = final_loss
                results['loss_decrease'] = loss_decrease
                results['best_node'] = node
                results['loss_history'] = loss_history
                results['success'] = loss_decrease > 0
                
        except Exception as e:
            print(f"  ✗ Error with lr={lr}: {e}")
            results['all_results'].append({
                'lr': lr,
                'error': str(e)
            })
    
    # Print summary
    print(f"\n{'-'*60}")
    if results['success']:
        print(f"✓ {node_name} PASSED")
        print(f"  Best LR: {results['best_lr']}")
        print(f"  Loss: {results['initial_loss']:.6f} → {results['final_loss']:.6f}")
        print(f"  Decrease: {results['loss_decrease']:.6f}")
    else:
        print(f"✗ {node_name} FAILED - No learning rate showed improvement")
    print(f"{'-'*60}")
    
    return results


def plot_node_surface(ax, node, node_name, lr, success=True, grid_size=20):
    """
    Generate 3D surface plot of the trained node on a given axis.
    
    Args:
        ax: Matplotlib 3D axis to plot on
        node: Trained node instance (or None if failed)
        node_name: Name of the node
        lr: Learning rate used (or None if failed)
        success: Whether training was successful
        grid_size: Resolution of the grid
    """
    if not success or node is None:
        # Plot empty surface for failed nodes
        ax.text(0.5, 0.5, 0.5, 'Training Failed', 
               horizontalalignment='center', verticalalignment='center',
               fontsize=14, color='red', weight='bold')
        ax.set_xlabel('$x_1$', fontsize=9)
        ax.set_ylabel('$x_2$', fontsize=9)
        ax.set_zlabel('Output', fontsize=9)
        ax.set_title(f'{node_name}\n(Failed)', fontsize=10, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        return
    
    # Create grid
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    inputs = np.stack([xx.flatten(), yy.flatten()], axis=1)
    
    # Get predictions with 3D input shape (batch_size, 1, input_dim)
    with torch.no_grad():
        inp = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1)  # (grid_size^2, 1, 2)
        out = node(inp)
        out = out.reshape(-1).cpu().numpy().reshape(grid_size, grid_size)
    
    # Plot surface
    ax.plot_surface(xx, yy, out, cmap='viridis', edgecolor='k', 
                    alpha=0.8, linewidth=0.2)
    
    # Plot ground truth points
    xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_outputs = xorc(xor_inputs[:, 0], xor_inputs[:, 1])
    
    for (xi, yi), zi in zip(xor_inputs, xor_outputs):
        # Pin needles
        ax.plot([xi, xi], [yi, yi], [0, zi], color='k', linewidth=2, zorder=10)
        # Points
        ax.scatter([xi], [yi], [zi], color='r', s=60, edgecolor='k', 
                  linewidth=1.5, zorder=11)
    
    # Labels and title
    ax.set_xlabel('$x_1$', fontsize=9)
    ax.set_ylabel('$x_2$', fontsize=9)
    ax.set_zlabel('Output', fontsize=9)
    title_color = 'green' if success else 'red'
    ax.set_title(f'{node_name}\n(LR={lr})', fontsize=10, color=title_color)
    ax.set_zlim(0, 1)


def plot_all_surfaces(all_results, save_path):
    """
    Plot all node surfaces in a single figure.
    
    Args:
        all_results: List of result dictionaries
        save_path: Path to save the plot
    """
    n_nodes = len(all_results)
    n_cols = 4
    n_rows = (n_nodes + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(16, 4 * n_rows))
    
    for idx, result in enumerate(all_results):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        
        plot_node_surface(
            ax,
            result.get('best_node'),
            result['node_name'],
            result.get('best_lr'),
            result.get('success', False)
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Combined surface plot saved: {save_path}")


def plot_loss_curves(all_results, save_path):
    """
    Plot loss curves for all nodes (successful and failed) in separate subplots.
    
    Args:
        all_results: List of result dictionaries
        save_path: Path to save the plot
    """
    # Separate successful and failed
    results_with_history = [r for r in all_results if r.get('loss_history') is not None]
    
    if not results_with_history:
        print("No loss histories to plot")
        return
    
    # Create subplots
    n_nodes = len(results_with_history)
    n_cols = 4
    n_rows = (n_nodes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_nodes > 1 else [axes]
    
    for idx, result in enumerate(results_with_history):
        ax = axes[idx]
        ax.plot(result['loss_history'], linewidth=2)
        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel('MSE Loss', fontsize=9)
        
        title_color = 'green' if result['success'] else 'red'
        status = 'Success' if result['success'] else 'Failed'
        ax.set_title(f"{result['node_name']}\n(LR={result['best_lr']}, {status})", 
                    fontsize=10, color=title_color)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.tick_params(labelsize=8)
    
    # Hide unused subplots
    for idx in range(len(results_with_history), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Individual loss curves saved: {save_path}")


def plot_combined_loss_curves(all_results, save_path):
    """
    Plot all loss curves in a single plot for direct comparison.
    
    Args:
        all_results: List of result dictionaries
        save_path: Path to save the plot
    """
    # Get results with history
    results_with_history = [r for r in all_results if r.get('loss_history') is not None]
    
    if not results_with_history:
        print("No loss histories to plot")
        return
    
    # Create single plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color palette for different nodes
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_with_history)))
    
    for idx, result in enumerate(results_with_history):
        label = f"{result['node_name']} (LR={result['best_lr']})"
        linestyle = '-' if result['success'] else '--'
        linewidth = 2 if result['success'] else 1.5
        alpha = 1.0 if result['success'] else 0.6
        
        ax.plot(result['loss_history'], 
               label=label, 
               color=colors[idx],
               linestyle=linestyle,
               linewidth=linewidth,
               alpha=alpha)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Training Loss Comparison Across All Nodes', fontsize=14, weight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9, ncol=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Combined loss comparison saved: {save_path}")


def main():
    """Main test function."""
    # Setup output directory first
    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"test_results_{timestamp}.log"
    logger = TeeLogger(str(log_file))
    sys.stdout = logger
    
    try:
        from difflut.registry import REGISTRY
        
        print("="*60)
        print("DiffLUT Node Training Test Suite")
        print("="*60)
        
        # Configuration
        EPOCHS = 50
        LEARNING_RATES = [0.1]
        N_SAMPLES = 2000
        
        print(f"\nOutput directory: {output_dir}")
        print(f"Log file: {log_file}")
        
        # Generate training data
        print(f"\nGenerating training data ({N_SAMPLES} samples)...")
        X_train, Y_train = generate_training_data(N_SAMPLES)
        
        # Get all registered nodes
        node_names = REGISTRY.list_nodes()
        print(f"\nFound {len(node_names)} registered nodes:")
        for name in node_names:
            print(f"  - {name}")
        
        # Test all nodes
        all_results = []
        
        for node_name in node_names:
            try:
                node_class = REGISTRY.get_node(node_name)
                result = test_node(
                    node_name, node_class, X_train, Y_train, 
                    LEARNING_RATES, EPOCHS
                )
                all_results.append(result)
                    
            except Exception as e:
                print(f"\n✗ CRITICAL ERROR testing {node_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    'node_name': node_name,
                    'success': False,
                    'error': str(e)
                })
        
        # Generate combined surface plot
        print("\nGenerating combined 3D surface plot...")
        surface_plot_path = output_dir / "all_nodes_surfaces.png"
        plot_all_surfaces(all_results, surface_plot_path)
        
        # Generate individual loss curves plot
        print("\nGenerating individual loss curves plot...")
        loss_plot_path = output_dir / "loss_curves_individual.png"
        plot_loss_curves(all_results, loss_plot_path)
        
        # Generate combined loss curves plot
        print("\nGenerating combined loss comparison plot...")
        combined_loss_plot_path = output_dir / "loss_curves_combined.png"
        plot_combined_loss_curves(all_results, combined_loss_plot_path)
        
        # Print final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        
        successful = [r for r in all_results if r['success']]
        failed = [r for r in all_results if not r['success']]
        
        print(f"\n✓ Passed: {len(successful)}/{len(all_results)} nodes")
        print(f"✗ Failed: {len(failed)}/{len(all_results)} nodes")
        
        if successful:
            print("\nSuccessful nodes:")
            for r in successful:
                print(f"  ✓ {r['node_name']:30s} "
                      f"(LR={r['best_lr']}, Loss: {r['initial_loss']:.4f}→{r['final_loss']:.4f})")
        
        if failed:
            print("\nFailed nodes:")
            for r in failed:
                error_msg = r.get('error', 'No improvement with any learning rate')
                print(f"  ✗ {r['node_name']:30s} ({error_msg})")
        
        print("\n" + "="*60)
        print(f"All plots saved to: {output_dir}")
        print(f"  - Combined surfaces: {surface_plot_path.name}")
        print(f"  - Individual loss curves: {loss_plot_path.name}")
        print(f"  - Combined loss comparison: {combined_loss_plot_path.name}")
        print("="*60)
        
        # Return exit code
        return 0 if len(failed) == 0 else 1
        
    finally:
        # Close the logger
        logger.close()
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
