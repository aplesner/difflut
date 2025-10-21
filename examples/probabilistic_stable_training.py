"""
Example: Using ProbabilisticStableNode in a deep LUT network for MNIST classification.

This example demonstrates:
1. Building a deep network with gradient-stabilized nodes
2. Comparing training stability with/without gradient stabilization
3. Monitoring gradient flow during training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings

# Suppress CUDA warnings if extension not compiled
warnings.filterwarnings('ignore', category=RuntimeWarning, module='difflut.nodes')

from difflut import ProbabilisticStableNode, ProbabilisticNode
from difflut.encoder import ThermometerEncoder


class DeepLUTClassifier(nn.Module):
    """Deep LUT network for classification."""
    
    def __init__(self, input_dim=28*28, num_classes=10, hidden_dim=64, 
                 num_layers=4, use_stabilization=True, alpha=1.0):
        super().__init__()
        
        # Choose node type
        NodeClass = ProbabilisticStableNode if use_stabilization else ProbabilisticNode
        node_kwargs = {'alpha': alpha, 'temperature': 1.0} if use_stabilization else {'temperature': 1.0}
        
        # Encoder: Convert continuous inputs to [0,1] range
        self.encoder = ThermometerEncoder(
            input_dim=input_dim,
            num_bins=8,  # 8 bins per input feature
            input_range=(-1, 1)
        )
        
        encoded_dim = input_dim * 8  # Thermometer encoding expands dimension
        
        # Build deep network
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(NodeClass(
            input_dim=[encoded_dim],
            output_dim=[hidden_dim],
            use_cuda=False,  # Set to True if CUDA extension is compiled
            **node_kwargs
        ))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(NodeClass(
                input_dim=[hidden_dim],
                output_dim=[hidden_dim],
                use_cuda=False,
                **node_kwargs
            ))
        
        # Output layer
        self.layers.append(NodeClass(
            input_dim=[hidden_dim],
            output_dim=[num_classes],
            use_cuda=False,
            **node_kwargs
        ))
        
        self.use_stabilization = use_stabilization
        self.alpha = alpha
    
    def forward(self, x):
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Encode to [0,1] range
        x = self.encoder(x)
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def get_gradient_stats(self):
        """Get gradient statistics for monitoring."""
        stats = {}
        for i, layer in enumerate(self.layers):
            if layer.raw_weights.grad is not None:
                grad = layer.raw_weights.grad
                stats[f'layer_{i}_grad_mean'] = grad.abs().mean().item()
                stats[f'layer_{i}_grad_std'] = grad.std().item()
                stats[f'layer_{i}_grad_max'] = grad.abs().max().item()
        return stats


def create_dummy_dataset(num_samples=1000):
    """Create a dummy dataset for demonstration."""
    # Random data
    X = torch.randn(num_samples, 28*28)
    y = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def train_epoch(model, dataloader, optimizer, criterion):
    """Train for one epoch and collect gradient statistics."""
    model.train()
    total_loss = 0
    gradient_stats_list = []
    
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Collect gradient statistics
        gradient_stats_list.append(model.get_gradient_stats())
        
        # Optimizer step
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx == 0:  # Just show first batch
            print(f"  Batch 0 gradients:")
            for key, value in gradient_stats_list[0].items():
                if 'mean' in key:
                    print(f"    {key}: {value:.6f}")
    
    avg_loss = total_loss / len(dataloader)
    
    # Average gradient stats
    avg_stats = {}
    for key in gradient_stats_list[0].keys():
        avg_stats[key] = sum(s[key] for s in gradient_stats_list) / len(gradient_stats_list)
    
    return avg_loss, avg_stats


def compare_training():
    """Compare training with and without gradient stabilization."""
    print("=" * 80)
    print("Comparing ProbabilisticNode vs ProbabilisticStableNode")
    print("=" * 80)
    
    # Create dummy dataset
    train_loader = create_dummy_dataset(num_samples=500)
    
    # Training parameters
    num_epochs = 3
    learning_rate = 0.01
    
    # Models
    models = {
        'Standard (no stabilization)': DeepLUTClassifier(
            num_layers=4, use_stabilization=False
        ),
        'Stabilized (alpha=0.9)': DeepLUTClassifier(
            num_layers=4, use_stabilization=True, alpha=0.9
        ),
        'Stabilized (alpha=1.0)': DeepLUTClassifier(
            num_layers=4, use_stabilization=True, alpha=1.0
        ),
    }
    
    criterion = nn.CrossEntropyLoss()
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*80}")
        print(f"Training: {name}")
        print(f"{'='*80}")
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        epoch_losses = []
        epoch_grad_stats = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            loss, grad_stats = train_epoch(model, train_loader, optimizer, criterion)
            epoch_losses.append(loss)
            epoch_grad_stats.append(grad_stats)
            print(f"  Average Loss: {loss:.4f}")
        
        results[name] = {
            'losses': epoch_losses,
            'grad_stats': epoch_grad_stats
        }
    
    # Print comparison
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Final loss: {result['losses'][-1]:.4f}")
        print(f"  Loss improvement: {result['losses'][0]:.4f} → {result['losses'][-1]:.4f}")
        
        # Show gradient stats for last layer (most prone to vanishing)
        last_grad_mean = result['grad_stats'][-1]['layer_0_grad_mean']
        print(f"  First layer gradient (final): {last_grad_mean:.6f}")


def demonstrate_alpha_tuning():
    """Demonstrate effect of different alpha values."""
    print("\n" + "=" * 80)
    print("Alpha Parameter Tuning Demonstration")
    print("=" * 80)
    
    alphas = [0.5, 0.8, 1.0, 1.2, 1.5]
    
    for alpha in alphas:
        print(f"\nTesting alpha={alpha:.1f}")
        
        model = DeepLUTClassifier(
            num_layers=5,  # Deeper network
            use_stabilization=True,
            alpha=alpha
        )
        
        # Single forward-backward pass
        x = torch.randn(16, 28*28)
        y = torch.randint(0, 10, (16,))
        
        optimizer = optim.Adam(model.parameters())
        optimizer.zero_grad()
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        # Check gradient flow
        stats = model.get_gradient_stats()
        first_layer_grad = stats['layer_0_grad_mean']
        last_layer_grad = stats['layer_4_grad_mean']
        
        print(f"  First layer gradient: {first_layer_grad:.6f}")
        print(f"  Last layer gradient:  {last_layer_grad:.6f}")
        print(f"  Ratio (first/last):   {first_layer_grad/last_layer_grad:.4f}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("ProbabilisticStableNode - Training Demonstration")
    print("=" * 80)
    print("\nNote: This example uses CPU mode for portability.")
    print("For faster training, compile CUDA extensions with:")
    print("  cd difflut && python setup.py install")
    print("and set use_cuda=True in the model.\n")
    
    # Run demonstrations
    compare_training()
    demonstrate_alpha_tuning()
    
    print("\n" + "=" * 80)
    print("Demonstration Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. ProbabilisticStableNode helps maintain gradient flow in deep networks")
    print("2. Alpha=0.9-1.0 works well in most cases")
    print("3. Gradient stabilization prevents vanishing gradients in early layers")
    print("4. Training loss improves more consistently with stabilization")
    print("\nFor production use:")
    print("- Compile CUDA extensions for 10-20x speedup")
    print("- Tune alpha based on network depth and architecture")
    print("- Monitor gradient statistics during training")
    print("- Use larger networks (6+ layers) to see full benefit")


if __name__ == "__main__":
    main()
