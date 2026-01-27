import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SUPPORTED_DATASETS = ['MNIST', 'CIFAR10']


def load_dataset(config):
    dataset_name = config['dataset']

    if dataset_name == 'MNIST':
        return load_mnist(
            config['data_path'],
            config['train_size'],
            config['val_size'],
            config['batch_size']
        )
    elif dataset_name == 'CIFAR10':
        return load_cifar10(
            config['data_path'],
            config['train_size'],
            config['val_size'],
            config['batch_size']
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {SUPPORTED_DATASETS}")


def load_mnist(data_path='./data', train_size=55000, val_size=5000, batch_size=60):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
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
    
    assert train_size + val_size == 60000, "Train + val must equal 60000 for MNIST"
    
    train_subset, val_subset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=1000,
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


def load_cifar10(data_path='./data', train_size=45000, val_size=5000, batch_size=60):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    full_train_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )

    assert train_size + val_size == 50000, "Train + val must equal 50000 for CIFAR-10"

    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=1000,
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


