"""
Pruning algorithms for the Lottery Ticket Hypothesis project.

This module implements:
1. Iterative pruning (main algorithm from the paper)
2. One-shot pruning 
3. Random reinitialization pruning (baseline comparison)

Author: Jerry (Tianrui Hu) (EE-411 Team)
Based on: "The Lottery Ticket Hypothesis" (Frankle & Carbin, 2019)
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Type
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
    'resnet20': {
        'prune_rate_conv': 0.2, # ResNet uses global pruning, typically 20%
        'prune_rate_fc': 0.0,   # ResNet does not prune FC layers in the paper
        'pruning_strategy': 'global', # Remember ResNet uses global pruning
        'warmup_strategy': 'linear_20k', # <--- Enable the logic written above
        'description': 'ResNet-20 LR=0.03 Warmup=20k'
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
    
    CRITICAL FIX: 
    1. Uses 'named_modules()' to strictly identify nn.Conv2d layers.
    2. Explicitly protects BatchNorm layers from being pruned.
    3. Excludes 'downsample' (shortcut) layers for stability in ResNet.
    """
    mask = {}
    
    # 1. Initialization: Set all masks to 1 (keep everything by default)
    #    This ensures BatchNorm, Biases, and FC layers are safe initially.
    for name, param in model.named_parameters():
        mask[name] = torch.ones_like(param)

    # Collect weights that we actually WANT to prune (only Conv2d weights)
    all_conv_weights = []
    # Store tuples of (full_param_name, parameter_tensor) for applying mask later
    conv_params_to_prune = []

    # 2. Iterate through modules to safely identify layer types
    for module_name, module in model.named_modules():
        
        # Check if it is a Convolutional layer
        if isinstance(module, nn.Conv2d):
            
            # Exclusion 1: Skip ResNet 'downsample' (shortcut) layers
            # Pruning shortcuts can cause dimension mismatch issues
            if 'downsample' in module_name:
                continue
            
            # Construct the full parameter name for the weight
            # e.g., module_name="layer1.0.conv1" -> param_name="layer1.0.conv1.weight"
            weight_name = f"{module_name}.weight"
            
            # Safety check: ensure this parameter actually exists
            if hasattr(module, 'weight') and module.weight is not None:
                param = module.weight
                
                # Add to collection for threshold calculation
                all_conv_weights.append(param.data.flatten())
                conv_params_to_prune.append((weight_name, param))

    # 3. Calculate Global Threshold
    if len(all_conv_weights) > 0 and prune_rate_conv > 0.0:
        # Concatenate all collected weights into one giant vector
        all_weights = torch.cat(all_conv_weights)
        
        # === Zero-Weight Fix (Iterative Pruning) ===
        # We must ignore weights that are already zero (pruned in previous rounds)
        # Otherwise, the quantile will be skewed by zeros.
        non_zero_weights = all_weights[torch.abs(all_weights) > 1e-8]
        
        if len(non_zero_weights) > 0:
            global_threshold = torch.quantile(torch.abs(non_zero_weights), prune_rate_conv)
        else:
            global_threshold = 0.0
            
        # 4. Apply Threshold
        for name, param in conv_params_to_prune:
            # Create binary mask: 1 if abs(weight) >= threshold, else 0
            mask[name] = (torch.abs(param.data) >= global_threshold).float()
            
    return mask


# ============================================================================
# Random Sparse Experiment (For Figure 1 Baseline)
# ============================================================================

def create_random_mask(
    model: nn.Module, 
    prune_rate: float
) -> Dict[str, torch.Tensor]:
    """
    Create a random pruning mask (randomly remove p% of weights).
    The structure is random, independent of weight magnitude.
    """
    mask = {}
    
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
        
        # Skip if prune_rate is 0 or invalid
        if prune_rate <= 0.0:
            mask[name] = torch.ones_like(param)
            continue
            
        # Generate random noise with same shape as param
        rand_tensor = torch.rand_like(param.data)
        
        # Calculate threshold to prune lowest p% of the random values
        threshold = torch.quantile(rand_tensor, prune_rate)
        
        # Create mask: 1 = keep, 0 = prune
        mask[name] = (rand_tensor >= threshold).float()
    
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
    verbose: bool = True,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    patience: int = 10,
    min_delta: float = 0.001
) -> Dict:
    
    model.train()
    
    # History tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'test_acc': [],
        'early_stop_epoch': None, # Which Epoch triggered (Figure 1)
        'early_stop_iter': None,  
        'early_stop_test_acc': None, 
        'early_stop_val_acc': None   
    }
    
    # Early Stop Variables
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_triggered = False
    
    # Calculate steps per epoch for iteration tracking
    steps_per_epoch = len(train_loader)
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        # 1. Training Loop
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if verbose and (epoch + 1) % 10 == 0 else train_loader
        
        for data, target in iterator:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if mask is not None:
                apply_mask(model, mask)
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # 2. Evaluation & Early Stop Logic
        current_val_loss = 0.0
        current_val_acc = 0.0 
        current_test_acc = 0.0
        

        if val_loader:
            current_val_loss, current_val_acc = evaluate_model(model, val_loader, device)
            history['val_loss'].append(current_val_loss)
            
        if test_loader:
            _, current_test_acc = evaluate_model(model, test_loader, device)
            history['test_acc'].append(current_test_acc)
            
        model.train()  # Switch back to train mode


        if val_loader and not early_stop_triggered:
            

            if epoch < 50:
                patience_counter = 0 

                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
            

            else:

                if current_val_loss < (best_val_loss - min_delta):
                    best_val_loss = current_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                

                if patience_counter >= patience:
                    early_stop_triggered = True
                    
                    history['early_stop_epoch'] = epoch + 1
                    history['early_stop_iter'] = (epoch + 1) * steps_per_epoch
                    history['early_stop_test_acc'] = current_test_acc 
                    history['early_stop_val_acc'] = current_val_acc
                    
                    if verbose:
                        print(f"   [Tracker] 🚩 Early Stop Condition at Iter {history['early_stop_iter']} "
                              f"(Epoch {epoch+1}, Test Acc: {current_test_acc:.2f}%)")


        # 4. Logging & Scheduler
        if verbose:
             print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {current_val_loss:.4f} | Val Acc: {current_val_acc:.2f}% | Test Acc: {current_test_acc:.2f}%")

        if scheduler is not None:
            scheduler.step()
            if (epoch + 1) in [56, 57, 71, 72] and verbose:
                current_lr = optimizer.param_groups[0]['lr']
                print(f" -> Scheduler Step! Current LR: {current_lr}")
    
    if not early_stop_triggered and val_loader:
        history['early_stop_epoch'] = epochs
        history['early_stop_iter'] = epochs * steps_per_epoch
        history['early_stop_test_acc'] = history['test_acc'][-1] if history['test_acc'] else 0.0
        
    return history


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

    # === New ===
    config['optimizer_kwargs'] = optimizer_kwargs
    config['epochs_per_round'] = epochs_per_round
    config['n_rounds'] = n_rounds
    
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
        if 'warmup_strategy' in config:
            print(f"Warmup strategy: {config['warmup_strategy']}")
        print("=" * 70)
    
    for round_idx in range(n_rounds):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Round {round_idx + 1}/{n_rounds}")
            print(f"{'='*70}")
        
        # Step 2: Train the model
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        criterion = nn.CrossEntropyLoss()

        # === NEW: Setup Scheduler for ResNet-18 ===
        # ResNet-18 requires specific LR decay at 20k and 25k iterations.
        # We calculate milestones dynamically based on epochs_per_round.
        scheduler = None

        # === Define Special Scheduler for Rate 0.03 + Warmup 20k ===
        if config.get('warmup_strategy') == 'linear_20k':
            # Assume total epochs = 86 (corresponding to 30k iters)
            # Warmup lasts until epoch 57 (corresponding to 20k iters)
            # Decay at epoch 71 (corresponding to 25k iters)


            warmup_epochs = int(epochs_per_round * 2 / 3)
            decay_epoch = int(epochs_per_round * 5 / 6)
            
            def lr_lambda(current_epoch, warmup_epochs=warmup_epochs, decay_epoch=decay_epoch):

                
                if current_epoch < warmup_epochs:
                    # Linear warmup: increase from 0 to 1.0 (i.e., 1.0 * base_lr 0.03)
                    # To avoid division by zero, add a small epsilon or start from step 1
                    return float(current_epoch + 1) / warmup_epochs
                elif current_epoch < decay_epoch:
                    # First decay interval: 20k - 25k iters
                    # LR should be 0.003, which is 0.1x relative to base 0.03
                    return 0.1
                else:
                    # Second decay interval: 25k+ iters
                    # LR should be 0.0003, which is 0.01x relative to base 0.03
                    return 0.01

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            
            if verbose:
                # print("Scheduler enabled: Linear Warmup to epoch 57, then decay.")
                print(f"Scheduler enabled: Linear Warmup to epoch {warmup_epochs}, then decay.")

        # ... Original ResNet scheduler without warmup ...
        elif 'resnet' in config.get('description', '').lower():
        # Check if we are running the ResNet-18/-20 config
        # if 'resnet' in config.get('description', '').lower():
            # Milestone 1: ~20k/30k iterations (2/3 of training)

            m1 = int(epochs_per_round * 2 / 3)
            # Milestone 2: ~25k/30k iterations (5/6 of training)

            m2 = int(epochs_per_round * 5 / 6)
            
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=[m1, m2], 
                gamma=0.1
            )
            
            if verbose:
                print(f"Scheduler enabled: MultiStepLR at epochs {m1} and {m2}")

        
        # Train with current mask (if any)
        history = train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=epochs_per_round,
            mask=current_mask,
            scheduler=scheduler,  
            test_loader=test_loader,  
            val_loader=val_loader,  
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
            'history': history,
            'mask': copy.deepcopy(current_mask)
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





def random_reinit_pruning(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer_class: type,
    optimizer_kwargs: Dict,
    device: torch.device,
    mask: Dict[str, torch.Tensor], # Pruning mask to apply, which is from iterative pruning
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