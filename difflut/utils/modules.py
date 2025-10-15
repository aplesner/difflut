import torch
import torch.nn as nn

class GroupSum(nn.Module):
    """
    Groups input features and sums them, dividing by tau.
    Fixed reshaping logic for proper grouping.
    """
    def __init__(self, k, tau=1, use_randperm=False):
        super().__init__()
        self.k = k  # Number of output groups
        self.tau = tau
        self.use_randperm = use_randperm
        
    def forward(self, x):
        # x shape: (batch_size, input_features)
        
        # Optionally permute
        if self.use_randperm:
            perm = torch.randperm(x.shape[-1], device=x.device)
            x = x[:, perm]
        
        # Pad if necessary
        if x.shape[-1] % self.k != 0:
            pad_size = self.k - (x.shape[-1] % self.k)
            x = nn.functional.pad(x, (0, pad_size))
        
        # Reshape to (batch_size, k groups, elements_per_group)
        group_size = x.shape[-1] // self.k
        x = x.view(x.shape[0], self.k, group_size)
        
        # Sum within each group and divide by tau
        # Result shape: (batch_size, k)
        return x.sum(dim=-1) / self.tau
