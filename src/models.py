"""
Neural network architectures for lottery ticket experiments.
"""

import torch
import torch.nn as nn


class LeNet300_100(nn.Module):
    """
    Fully-connected network for MNIST.
    Architecture from LeCun et al. (1998) as used in lottery ticket paper.
    
    Structure:
        Input (784) -> FC(300) -> ReLU -> FC(100) -> ReLU -> FC(10)
    
    Total parameters: ~266K
    """
    def __init__(self, num_classes=10):
        super(LeNet300_100, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Layer 1
        x = self.relu(self.fc1(x))
        
        # Layer 2
        x = self.relu(self.fc2(x))
        
        # Output layer (no activation - handled by loss function)
        x = self.fc3(x)
        
        return x
    
    def count_parameters(self):
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_active_parameters(self, mask_dict=None):
        """Count number of active (non-pruned) parameters"""
        if mask_dict is None:
            return self.count_parameters()
        
        total = 0
        for name, param in self.named_parameters():
            if name in mask_dict:
                total += (mask_dict[name] != 0).sum().item()
            else:
                total += param.numel()
        
        return total


def create_model(model_name, num_classes=10, device='cpu'):
    """
    Factory function to create models.
    
    Args:
        model_name: Name of the model ('LeNet300_100')
        num_classes: Number of output classes
        device: Device to put model on
    
    Returns:
        model: Initialized model on specified device
    """
    if model_name == 'LeNet300_100':
        model = LeNet300_100(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Initialize with Glorot (Xavier) initialization
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    return model.to(device)


def get_model_info(model):
    """
    Get summary information about the model.
    
    Returns:
        dict with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count parameters per layer
    layer_params = {}
    for name, param in model.named_parameters():
        layer_params[name] = param.numel()
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'layer_parameters': layer_params,
        'model_size_mb': total_params * 4 / (1024 ** 2)  # Assuming float32
    }