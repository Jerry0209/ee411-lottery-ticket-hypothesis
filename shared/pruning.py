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
    val_loader: Optional[torch.utils.data.DataLoader],
    optimizer_class: type,
    optimizer_kwargs: Dict,
    device: torch.device,
    config: Optional[Dict] = None,
    n_rounds: int = 15,
    epochs_per_round: int = 20,
    verbose: bool = True
) -> List[Dict]:

    # Default Conv-6 config (Figure 2 / paper table style)
    if config is None:
        config = {
            "prune_rate_conv": 0.15,        # Conv layers: 15% per round
            "prune_rate_fc": 0.20,          # FC layers: 20% per round
            "pruning_strategy": "layerwise",
            "description": "Conv-6 for CIFAR-10"
        }

    # Store run metadata (optional)
    config["optimizer_kwargs"] = optimizer_kwargs
    config["epochs_per_round"] = epochs_per_round
    config["n_rounds"] = n_rounds

    model = model.to(device)

    # Save initial weights θ0
    initial_weights = copy.deepcopy(model.state_dict())
    initial_params = count_parameters(model)

    results: List[Dict] = []
    current_mask = None

    if verbose:
        print("=" * 70)
        print("Starting Iterative Pruning (Conv-6)")
        print(f"Initial parameters: {initial_params:,}")
        print(f"Pruning strategy: {config['pruning_strategy']}")
        print(f"Conv prune rate: {config['prune_rate_conv']:.1%}")
        print(f"FC prune rate: {config['prune_rate_fc']:.1%}")
        print(f"Optimizer: {optimizer_class.__name__} {optimizer_kwargs}")
        print("=" * 70)

    for round_idx in range(n_rounds):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Round {round_idx + 1}/{n_rounds}")
            print(f"{'='*70}")

        # Step 2: Train (Conv-6: no special scheduler/warmup)
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        criterion = nn.CrossEntropyLoss()
        scheduler = None  # <- keep None for Conv-6

        history = train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=epochs_per_round,
            mask=current_mask,
            scheduler=scheduler,      # stays None
            test_loader=test_loader,  # keep if your train_model logs test/val
            val_loader=val_loader,
            verbose=verbose
        )

        # Evaluate after training
        test_loss, test_accuracy = evaluate_model(model, test_loader, device)

        # Step 3: Prune -> create new mask
        if config["pruning_strategy"] == "layerwise":
            new_mask = create_layerwise_mask(
                model,
                config["prune_rate_conv"],
                config["prune_rate_fc"]
            )
        elif config["pruning_strategy"] == "global":
            # Conv-6 is normally layerwise; but keep this for completeness
            new_mask = create_global_mask(model, config["prune_rate_conv"])
        else:
            raise ValueError(f"Unknown pruning strategy: {config['pruning_strategy']}")

        # Accumulate masks (once pruned, always pruned)
        if current_mask is not None:
            for name in new_mask:
                if name in current_mask:
                    new_mask[name] = new_mask[name] * current_mask[name]

        current_mask = new_mask

        # Compute remaining weights (percent weights remaining)
        remaining_params = count_nonzero_parameters(model, current_mask)
        remaining_pct = 100.0 * remaining_params / initial_params

        # Step 4: Rewind to θ0 and apply mask
        model.load_state_dict(initial_weights)
        apply_mask(model, current_mask)

        result = {
            "round": round_idx + 1,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "remaining_params": remaining_params,
            "remaining_params_pct": remaining_pct,
            "history": history,
            "mask": copy.deepcopy(current_mask),
        }
        results.append(result)

        if verbose:
            print(f"\nRound {round_idx + 1} Results:")
            print(f"  Test Accuracy: {test_accuracy:.2f}%")
            print(f"  Remaining Parameters: {remaining_params:,} ({remaining_pct:.2f}%)")

    if verbose:
        print(f"\n{'='*70}")
        print("Iterative Pruning Complete (Conv-6)!")
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
    config: Optional[Dict] = None,  # [Added] To pass scheduler config
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    epochs: int = 50,
    verbose: bool = True
) -> Dict:
    """
    Random reinitialization pruning (Control Experiment).
    Re-initializes weights randomly while keeping the pruning mask structure.
    """
    # Use default config if none provided (prevents errors if key is missing)
    if config is None:
        config = {}

    model = model.to(device)
    
    # 0. Calculate total parameters (denominator for sparsity calculation)
    total_params = sum(p.numel() for p in model.parameters())

    # 1. Randomly reinitialize (Using Xavier/Glorot for ResNet/Conv layers)
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    # 2. Apply Mask (Enforce structure)
    apply_mask(model, mask)
    
    # Calculate sparsity stats
    remaining_params = count_nonzero_parameters(model, mask)
    remaining_pct = 100.0 * remaining_params / total_params

    if verbose:
        print("\n" + "=" * 70)
        print(f"Starting Random Reinit Control Experiment")
        print(f"Sparsity Level: {remaining_pct:.2f}% weights remaining")
        print(f"Pruning strategy: {config['pruning_strategy']}")
        print(f"Conv prune rate: {config['prune_rate_conv']:.1%}")
        print(f"FC prune rate: {config['prune_rate_fc']:.1%}")
        if 'warmup_strategy' in config:
            print(f"Warmup strategy: {config['warmup_strategy']}")
        print("=" * 70)
    
    # 3. Setup Optimizer
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss()

    # 4. Setup Scheduler (Synced with iterative_pruning)
    scheduler = None
    
    if epochs > 0:
        # === Strategy 1: Rate 0.03 + Warmup 20k ===
        if config.get('warmup_strategy') == 'linear_20k':
            # Dynamic milestones: 20k (~2/3) and 25k (~5/6) iterations
            warmup_epochs = int(epochs * 2 / 3)
            decay_epoch = int(epochs * 5 / 6)
            
            def lr_lambda(current_epoch, warmup_epochs=warmup_epochs, decay_epoch=decay_epoch):
                if current_epoch < warmup_epochs:
                    # Linear warmup: 0 -> 1.0 (relative to base LR)
                    return float(current_epoch + 1) / warmup_epochs
                elif current_epoch < decay_epoch:
                    # First decay (20k-25k): 0.1x
                    return 0.1
                else:
                    # Second decay (25k+): 0.01x
                    return 0.01

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            
            if verbose:
                print(f"Scheduler enabled: Linear Warmup to epoch {warmup_epochs}, then decay.")

        # === Strategy 2: Standard ResNet (MultiStepLR) ===
        # Check description or fallback to standard logic
        elif 'resnet' in config.get('description', '').lower():
            # Milestone 1: ~20k iters (2/3 of training)
            m1 = int(epochs * 2 / 3)
            # Milestone 2: ~25k iters (5/6 of training)
            m2 = int(epochs * 5 / 6)
            
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=[m1, m2], 
                gamma=0.1
            )
            
            if verbose:
                print(f"Scheduler enabled: MultiStepLR at epochs {m1} and {m2}")

    # 5. Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs,
        mask=mask,
        scheduler=scheduler,
        test_loader=test_loader,
        val_loader=val_loader,
        verbose=verbose
    )
    
    # 6. Evaluate Final Performance
    test_loss, test_accuracy = evaluate_model(model, test_loader, device)
    
    if verbose:
        print(f"\nRandom Reinit Result:")
        print(f"  Test Accuracy: {test_accuracy:.2f}%")
        print(f"  Remaining Params: {remaining_pct:.2f}%")
        print("=" * 70 + "\n")
    
    # 7. Return Data
    return {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'remaining_params': remaining_params,
        'remaining_params_pct': remaining_pct,
        'history': history,
        'mask': None  # Not saving mask to conserve memory
    }

    

def random_sparse_pruning(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer_class: type,
    optimizer_kwargs: Dict,
    device: torch.device,
    sparsity_levels: List[float],
    n_trials: int = 3,
    epochs: int = 86,
    verbose: bool = True
) -> Dict[float, List[Dict]]:
    """
    Run Random Sparse Network experiment (Baseline for Figure 1).
    """
    results = {} # format: {sparsity_float: [trial_dict_1, trial_dict_2...]}
    
    model = model.to(device)
    
    # 1. Get Initial Param Count (Denominator for percentage calculation)
    # This ensures consistency with iterative_pruning
    initial_params = count_parameters(model)
    
    if verbose:
        print("\n" + "="*70)
        print("RUNNING EXPERIMENT: RANDOM SPARSE NETWORKS")
        print(f"Sparsity Levels: {sparsity_levels}")
        print(f"Trials per level: {n_trials}")
        print(f"Total Params: {initial_params:,}")
        print("="*70)

    for sparsity in sparsity_levels:
        results[sparsity] = []
        
        for trial in range(n_trials):
            if verbose:
                print(f"\n[Sparsity {sparsity:.2f} | Trial {trial+1}/{n_trials}]")
            
            # 2. Randomly Reinitialize Model (Weights)
            # Must re-init weights for "Random Sparse" baseline (Control Group)
            def init_weights(m):
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            model.apply(init_weights)
            
            # 3. Create & Apply Random Mask (Structure)
            # Pruning rate = sparsity (e.g., 0.2 means remove 20%)
            mask = create_random_mask(model, prune_rate=sparsity)
            apply_mask(model, mask)
            
            # 4. Calculate Stats (Strictly mimicking iterative_pruning)
            remaining_params = count_nonzero_parameters(model, mask)
            remaining_pct = 100.0 * remaining_params / initial_params
            
            if verbose:
                 print(f" -> Structure created. Remaining: {remaining_params:,} ({remaining_pct:.2f}%)")

            # 5. Setup Optimizer & Scheduler (ResNet specific)
            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
            criterion = nn.CrossEntropyLoss()
            
            scheduler = None
            # ResNet-18/20 standard milestones (approx 2/3 and 5/6 of training)
            m1 = int(epochs * 0.66)
            m2 = int(epochs * 0.83)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[m1, m2], gamma=0.1
            )
            
            # 6. Train with Early Stop Tracking
            # The history dict returned here contains 'early_stop_iter' etc.
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epochs=epochs,
                mask=mask,
                scheduler=scheduler,
                verbose=verbose
            )
            
            # 7. Collect Result (Format aligned with iterative_pruning)
            # using the final accuracy from history for consistency
            final_test_acc = history['test_acc'][-1] if history['test_acc'] else 0.0
            
            trial_result = {
                # Standard Keys (Match iterative_pruning)
                'round': f"Sparsity {sparsity:.2f}", # Placeholder
                'test_accuracy': final_test_acc,
                'remaining_params': remaining_params,
                'remaining_params_pct': remaining_pct, # Matches X-axis logic
                'history': history,
                # 'mask': mask, # Optional: Don't save mask to save disk space for random trials
                
                # Figure 1 Specific Keys (extracted from history)
                'early_stop_iter': history['early_stop_iter'],
                'early_stop_test_acc': history['early_stop_test_acc'],
                'early_stop_epoch': history['early_stop_epoch']
            }
            
            results[sparsity].append(trial_result)
            
            if verbose:
                print(f" -> Done. Early Stop Iter: {trial_result['early_stop_iter']} | "
                      f"Stop Acc: {trial_result['early_stop_test_acc']:.2f}% | "
                      f"Final Acc: {final_test_acc:.2f}%")
                
    return results
