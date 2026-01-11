"""
Data loading utilities for lottery ticket experiments.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def load_mnist(data_path='./data', train_size=55000, val_size=5000, batch_size=60):
    """
    Load MNIST dataset with train/val/test splits as used in paper.
    
    Paper uses:
    - Training: 55,000 examples
    - Validation: 5,000 examples
    - Test: 10,000 examples
    
    Args:
        data_path: Path to download/load data
        train_size: Number of training examples
        val_size: Number of validation examples
        batch_size: Batch size for dataloaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Normalization values from MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    full_train_dataset = datasets.MNIST(
        root=data_path, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_path, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Split training into train and validation
    assert train_size + val_size == 60000, "Train + val must equal 60000 for MNIST"
    
    train_subset, val_subset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility, increase if you have multiple cores
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=1000,  # Larger batch for validation (faster)
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1000, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader


def get_dataset_info(train_loader, val_loader, test_loader):
    """
    Get information about the dataset.
    
    Returns:
        dict with dataset statistics
    """
    return {
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'test_samples': len(test_loader.dataset),
        'train_batches': len(train_loader),
        'val_batches': len(val_loader),
        'test_batches': len(test_loader),
        'input_shape': (1, 28, 28),
        'num_classes': 10
    }