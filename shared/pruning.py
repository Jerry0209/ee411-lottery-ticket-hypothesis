"""
Pruning algorithms for the Lottery Ticket Hypothesis project.

This module implements:
1. Iterative pruning (main algorithm from the paper)
2. One-shot pruning 
3. Random reinitialization pruning (baseline comparison)

Author: Jerry (EE-411 Team)
Based on: "The Lottery Ticket Hypothesis" (Frankle & Carbin, 2019)
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


# ============================================================================
# Pruning Configuration Presets (from paper Figure 2)
# ============================================================================

PRUNING_CONFIGS = {
    'lenet': {
        'prune_rate_conv': 0.0,      # No conv layers
        'prune_rate_fc': 0.2,         # FC layers: 20%
        'pruning_strategy': 'layerwise',
        'description': 'LeNet-300-100 for MNIST'
    },
    'conv2': {
        'prune_rate_conv': 0.1,       # Conv layers: 10%
        'prune_rate_fc': 0.2,          # FC layers: 20%
        'pruning_strategy': 'layerwise',
        'description': 'Conv-2 for CIFAR-10'
    },
    'conv4': {
        'prune_rate_conv': 0.1,       # Conv layers: 10%
        'prune_rate_fc': 0.2,          # FC layers: 20%
        'pruning_strategy': 'layerwise',
        'description': 'Conv-4 for CIFAR-10'
    },
    'conv6': {
        'prune_rate_conv': 0.15,      # Conv layers: 15%
        'prune_rate_fc': 0.2,          # FC layers: 20%
        'pruning_strategy': 'layerwise',
        'description': 'Conv-6 for CIFAR-10'
    },
    'resnet18': {
        'prune_rate_conv': 0.2,       # Conv layers: 20%
        'prune_rate_fc': 0.0,          # Don't prune FC (too few params)
        'pruning_strategy': 'global',  # Use global pruning for deep networks
        'description': 'ResNet-18 for CIFAR-10'
    },
}


def get_pruning_config(model_name: str) -> Dict:
    """
    Get recommended pruning configuration for a specific model.
    
    Args:
        model_name: Name of the model ('lenet', 'conv2', 'conv4', 'conv6', 'resnet18')
    
    Returns:
        Dictionary with pruning configuration
    
    Example:
        >>> config = get_pruning_config('conv2')
        >>> print(config)
        {'prune_rate_conv': 0.1, 'prune_rate_fc': 0.2, ...}
    """
    model_name = model_name.lower()
    if model_name not in PRUNING_CONFIGS:
        print(f"Warning: Model '{model_name}' not found. Using 'conv2' config as default.")
        return PRUNING_CONFIGS['conv2'].copy()
    
    return PRUNING_CONFIGS[model_name].copy()


# ============================================================================
# Helper Functions
# ============================================================================

def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """Count total parameters in a model."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: nn.Module, mask: Optional[Dict] = None) -> int:
    """
    Count non-zero parameters (considering mask if provided).
    
    Args:
        model: PyTorch model
        mask: Optional mask dictionary from pruning
    
    Returns:
        Number of non-zero parameters
    """
    if mask is None:
        return sum((p != 0).sum().item() for p in model.parameters())
    
    total = 0
    for name, param in model.named_parameters():
        if name in mask:
            total += mask[name].sum().item()
        else:
            total += (param != 0).sum().item()
    
    return total


def is_conv_layer(name: str, param: torch.Tensor) -> bool:
    """
    Check if a parameter belongs to a convolutional layer.
    
    Args:
        name: Parameter name
        param: Parameter tensor
    
    Returns:
        True if conv layer, False otherwise
    """
    # Check by name
    if 'conv' in name.lower():
        return True
    
    # Check by tensor shape (conv weights are 4D: [out_ch, in_ch, h, w])
    if len(param.shape) == 4:
        return True
    
    return False


# ============================================================================
# Mask Creation Functions
# ============================================================================

def create_layerwise_mask(
    model: nn.Module, 
    prune_rate_conv: float, 
    prune_rate_fc: float
) -> Dict[str, torch.Tensor]:
    """
    Create pruning mask using layer-wise strategy.
    Each layer is pruned independently at its specified rate.
    
    Args:
        model: PyTorch model
        prune_rate_conv: Pruning rate for convolutional layers (0.0-1.0)
        prune_rate_fc: Pruning rate for fully-connected layers (0.0-1.0)
    
    Returns:
        Dictionary mapping parameter names to binary masks
    """
    mask = {}
    
    for name, param in model.named_parameters():
        if 'weight' not in name:
            # Don't prune biases
            continue
        
        # Determine pruning rate based on layer type
        if is_conv_layer(name, param):
            prune_rate = prune_rate_conv
        else:
            prune_rate = prune_rate_fc
        
        # Skip if prune_rate is 0
        if prune_rate == 0.0:
            mask[name] = torch.ones_like(param)
            continue
        
        # Calculate threshold (prune_rate percentile)
        threshold = torch.quantile(torch.abs(param.data), prune_rate)
        
        # Create mask: 1 = keep, 0 = prune
        mask[name] = (torch.abs(param.data) >= threshold).float()
    
    return mask


def create_global_mask(
    model: nn.Module, 
    prune_rate_conv: float
) -> Dict[str, torch.Tensor]:
    """
    Create pruning mask using global strategy.
    All convolutional layers are pruned together based on global magnitude.
    
    This is recommended for deep networks (ResNet, VGG) where layer sizes
    vary significantly. It prevents small layers from becoming bottlenecks.
    
    Args:
        model: PyTorch model
        prune_rate_conv: Global pruning rate for all conv layers
    
    Returns:
        Dictionary mapping parameter names to binary masks
    """
    mask = {}
    
    # Collect all conv layer weights
    all_conv_weights = []
    conv_params = []
    
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
        
        if is_conv_layer(name, param):
            all_conv_weights.append(param.data.flatten())
            conv_params.append((name, param))
        else:
            # For non-conv layers, don't prune (typical for ResNet/VGG)
            mask[name] = torch.ones_like(param)
    
    # Calculate global threshold across all conv layers
    if len(all_conv_weights) > 0 and prune_rate_conv > 0.0:
        all_weights = torch.cat(all_conv_weights)
        global_threshold = torch.quantile(torch.abs(all_weights), prune_rate_conv)
        
        # Apply global threshold to each conv layer
        for name, param in conv_params:
            mask[name] = (torch.abs(param.data) >= global_threshold).float()
    else:
        # No pruning
        for name, param in conv_params:
            mask[name] = torch.ones_like(param)
    
    return mask


def apply_mask(model: nn.Module, mask: Dict[str, torch.Tensor]):
    """
    Apply pruning mask to model parameters.
    
    Args:
        model: PyTorch model
        mask: Dictionary of masks from create_*_mask functions
    """
    for name, param in model.named_parameters():
        if name in mask:
            param.data *= mask[name]


# ============================================================================
# Training Helper
# ============================================================================

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 1,
    mask: Optional[Dict] = None,
    verbose: bool = True
) -> List[float]:
    """
    Train model for specified epochs.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epochs: Number of epochs
        mask: Optional pruning mask to apply after each step
        verbose: Whether to print progress
    
    Returns:
        List of average losses per epoch
    """
    model.train()
    losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else train_loader
        
        for batch_idx, (data, target) in enumerate(iterator):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Apply mask after gradient step (if pruned)
            if mask is not None:
                apply_mask(model, mask)
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return losses


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
    
    Returns:
        Tuple of (test_loss, test_accuracy)
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Loss
            loss = nn.functional.cross_entropy(output, target, reduction='sum')
            test_loss += loss.item()
            
            # Accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= total
    test_accuracy = 100.0 * correct / total
    
    return test_loss, test_accuracy


# ============================================================================
# Main Pruning Algorithms
# ============================================================================

def iterative_pruning(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer_class: type,
    optimizer_kwargs: Dict,
    device: torch.device,
    config: Optional[Dict] = None,
    n_rounds: int = 15,
    epochs_per_round: int = 20,
    verbose: bool = True
) -> List[Dict]:
    """
    Iterative pruning algorithm (main algorithm from the paper).
    
    Algorithm:
    1. Randomly initialize network f(x; θ₀)
    2. Train for j iterations → parameters θⱼ
    3. Prune p% of parameters (lowest magnitude) → create mask m
    4. Reset remaining parameters to θ₀ → winning ticket f(x; m ⊙ θ₀)
    5. Repeat steps 2-4 for n_rounds
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer_class: Optimizer class (e.g., torch.optim.Adam)
        optimizer_kwargs: Optimizer arguments (e.g., {'lr': 0.001})
        device: Device to train on
        config: Pruning configuration (if None, uses default)
        n_rounds: Number of pruning iterations
        epochs_per_round: Training epochs per round
        verbose: Whether to print progress
    
    Returns:
        List of dictionaries containing results for each round
    
    Example:
        >>> config = get_pruning_config('conv2')
        >>> results = iterative_pruning(
        ...     model=Conv2(),
        ...     train_loader=train_loader,
        ...     test_loader=test_loader,
        ...     optimizer_class=torch.optim.Adam,
        ...     optimizer_kwargs={'lr': 2e-4},
        ...     device=device,
        ...     config=config,
        ...     n_rounds=10
        ... )
    """
    # Default config
    if config is None:
        config = {
            'prune_rate_conv': 0.1,
            'prune_rate_fc': 0.2,
            'pruning_strategy': 'layerwise'
        }
    
    model = model.to(device)
    
    # Step 1: Save initial weights θ₀
    initial_weights = copy.deepcopy(model.state_dict())
    initial_params = count_parameters(model)
    
    results = []
    current_mask = None
    
    if verbose:
        print("=" * 70)
        print("Starting Iterative Pruning")
        print(f"Initial parameters: {initial_params:,}")
        print(f"Pruning strategy: {config['pruning_strategy']}")
        print(f"Conv prune rate: {config['prune_rate_conv']:.1%}")
        print(f"FC prune rate: {config['prune_rate_fc']:.1%}")
        print("=" * 70)
    
    for round_idx in range(n_rounds):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Round {round_idx + 1}/{n_rounds}")
            print(f"{'='*70}")
        
        # Step 2: Train the model
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=epochs_per_round,
            mask=current_mask,
            verbose=verbose
        )
        
        # Evaluate
        test_loss, test_accuracy = evaluate_model(model, test_loader, device)
        
        # Step 3: Prune (create mask)
        if config['pruning_strategy'] == 'layerwise':
            new_mask = create_layerwise_mask(
                model,
                config['prune_rate_conv'],
                config['prune_rate_fc']
            )
        elif config['pruning_strategy'] == 'global':
            new_mask = create_global_mask(
                model,
                config['prune_rate_conv']
            )
        else:
            raise ValueError(f"Unknown pruning strategy: {config['pruning_strategy']}")
        
        # Combine with previous mask (if exists)
        if current_mask is not None:
            for name in new_mask:
                if name in current_mask:
                    new_mask[name] = new_mask[name] * current_mask[name]
        
        current_mask = new_mask
        
        # Count remaining parameters
        remaining_params = count_nonzero_parameters(model, current_mask)
        remaining_pct = 100.0 * remaining_params / initial_params
        
        # Step 4: Reset to initial weights θ₀
        model.load_state_dict(initial_weights)
        apply_mask(model, current_mask)
        
        # Record results
        result = {
            'round': round_idx + 1,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'remaining_params': remaining_params,
            'remaining_params_pct': remaining_pct,
            'train_losses': train_losses
        }
        results.append(result)
        
        if verbose:
            print(f"\nRound {round_idx + 1} Results:")
            print(f"  Test Accuracy: {test_accuracy:.2f}%")
            print(f"  Remaining Parameters: {remaining_params:,} ({remaining_pct:.2f}%)")
    
    if verbose:
        print(f"\n{'='*70}")
        print("Iterative Pruning Complete!")
        print(f"{'='*70}")
    
    return results


def one_shot_pruning(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer_class: type,
    optimizer_kwargs: Dict,
    device: torch.device,
    target_sparsity: float = 0.8,
    epochs: int = 50,
    verbose: bool = True
) -> Dict:
    """
    One-shot pruning: Train once, prune once, retrain.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer_class: Optimizer class
        optimizer_kwargs: Optimizer arguments
        device: Device to train on
        target_sparsity: Target sparsity (0.8 = remove 80% of weights)
        epochs: Training epochs
        verbose: Whether to print progress
    
    Returns:
        Dictionary with results
    """
    model = model.to(device)
    initial_weights = copy.deepcopy(model.state_dict())
    
    if verbose:
        print("One-Shot Pruning: Training initial model...")
    
    # Train initial model
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, optimizer, criterion, device, epochs, verbose=verbose)
    
    # Prune all weights globally to target sparsity
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            all_weights.append(param.data.flatten())
    
    all_weights = torch.cat(all_weights)
    threshold = torch.quantile(torch.abs(all_weights), target_sparsity)
    
    mask = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask[name] = (torch.abs(param.data) >= threshold).float()
    
    # Reset and apply mask
    model.load_state_dict(initial_weights)
    apply_mask(model, mask)
    
    # Evaluate
    test_loss, test_accuracy = evaluate_model(model, test_loader, device)
    
    return {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'remaining_params': count_nonzero_parameters(model, mask),
        'mask': mask
    }


def random_reinit_pruning(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer_class: type,
    optimizer_kwargs: Dict,
    device: torch.device,
    mask: Dict[str, torch.Tensor],
    epochs: int = 50,
    verbose: bool = True
) -> Dict:
    """
    Random reinitialization pruning (baseline for comparison).
    
    Instead of resetting to original θ₀, randomly reinitialize.
    This should perform worse than iterative pruning (winning ticket).
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer_class: Optimizer class
        optimizer_kwargs: Optimizer arguments
        device: Device to train on
        mask: Pruning mask from iterative pruning
        epochs: Training epochs
        verbose: Whether to print progress
    
    Returns:
        Dictionary with results
    """
    model = model.to(device)
    
    # Randomly reinitialize
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    apply_mask(model, mask)
    
    if verbose:
        print("Random Reinitialization Pruning: Training...")
    
    # Train
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, optimizer, criterion, device, epochs, mask, verbose)
    
    # Evaluate
    test_loss, test_accuracy = evaluate_model(model, test_loader, device)
    
    return {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'remaining_params': count_nonzero_parameters(model, mask)
    }
