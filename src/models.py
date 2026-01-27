import torch
import torch.nn as nn

SUPPORTED_MODELS = ['LeNet300_100']


class LeNet300_100(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet300_100, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_active_parameters(self, mask_dict=None):
        if mask_dict is None:
            return self.count_parameters()
        
        total = 0
        for name, param in self.named_parameters():
            if name in mask_dict:
                total += (mask_dict[name] != 0).sum().item()
            else:
                total += param.numel()
        
        return total


def get_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    layer_params = {}
    for name, param in model.named_parameters():
        layer_params[name] = param.numel()
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'layer_parameters': layer_params,
        'model_size_mb': total_params * 4 / (1024 ** 2)
    }