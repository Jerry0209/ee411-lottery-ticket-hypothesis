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


# def create_global_mask(
#     model: nn.Module, 
#     prune_rate_conv: float
# ) -> Dict[str, torch.Tensor]:
#     """
#     Create pruning mask using global strategy.
#     All convolutional layers are pruned together based on global magnitude.
    
#     This is recommended for deep networks (ResNet, VGG) where layer sizes
#     vary significantly. It prevents small layers from becoming bottlenecks.
    
#     Args:
#         model: PyTorch model
#         prune_rate_conv: Global pruning rate for all conv layers
    
#     Returns:
#         Dictionary mapping parameter names to binary masks
#     """
#     mask = {}
    
#     # Collect all conv layer weights
#     all_conv_weights = []
#     conv_params = []
    
#     for name, param in model.named_parameters():
#         if 'weight' not in name:
#             continue
        
#         # if is_conv_layer(name, param) :
#         # New! Usually the shortcut layer named 'downsample' in ResNet should not be pruned
#         if is_conv_layer(name, param) and 'downsample' not in name:
#             all_conv_weights.append(param.data.flatten())
#             conv_params.append((name, param))
#         else:
#             # For non-conv layers, don't prune (typical for ResNet/VGG)
#             mask[name] = torch.ones_like(param)
    
#     # Calculate global threshold across all conv layers
#     if len(all_conv_weights) > 0 and prune_rate_conv > 0.0:
#         all_weights = torch.cat(all_conv_weights)
        
#         # global_threshold = torch.quantile(torch.abs(all_weights), prune_rate_conv)

#         # === Fix Start ===
#         # Only count non-zero weights for threshold calculation
#         # non_zero_weights = all_weights[all_weights != 0] 
#         non_zero_weights = all_weights[torch.abs(all_weights) > 1e-8]
        
#         if len(non_zero_weights) > 0:
#             global_threshold = torch.quantile(torch.abs(non_zero_weights), prune_rate_conv)
#         else:
#             global_threshold = 0.0
#         # === Fixed End ===
        
#         # Apply global threshold to each conv layer
#         for name, param in conv_params:
#             mask[name] = (torch.abs(param.data) >= global_threshold).float()
#     else:
#         # No pruning
#         for name, param in conv_params:
#             mask[name] = torch.ones_like(param)
    
#     return mask


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

# def train_model(
#     model: nn.Module,
#     train_loader: torch.utils.data.DataLoader,
#     optimizer: torch.optim.Optimizer,
#     criterion: nn.Module,
#     device: torch.device,
#     epochs: int = 1,
#     mask: Optional[Dict] = None,
#     verbose: bool = True,
#     scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, # New !
#     test_loader=None # New !
# ) -> List[float]:
#     """
#     Train model for specified epochs.
    
#     Args:
#         model: PyTorch model
#         train_loader: Training data loader
#         optimizer: Optimizer
#         criterion: Loss function
#         device: Device to train on
#         epochs: Number of epochs
#         mask: Optional pruning mask to apply after each step
#         verbose: Whether to print progress
    
#     Returns:
#         List of average losses per epoch
#     """
#     model.train()
#     # losses = []
#     history = {
#         'train_loss': [],
#         'test_acc': []  
#     }
    
#     for epoch in range(epochs):
#         running_loss = 0.0
        
#         iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if verbose and (epoch + 1) % 10 == 0 else train_loader
        
#         for batch_idx, (data, target) in enumerate(iterator):
#             data, target = data.to(device), target.to(device)
            
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
            
#             # Apply mask after gradient step (if pruned)
#             if mask is not None:
#                 apply_mask(model, mask)
            
#             running_loss += loss.item()
        
#         avg_loss = running_loss / len(train_loader)
#         history['train_loss'].append(avg_loss)

#         # --- Validation Step (New!) ---
#         if test_loader is not None:
#             test_loss, test_acc = evaluate_model(model, test_loader, device)
#             history['test_acc'].append(test_acc)
#             model.train() # Set back to train mode
#         else:
#             test_acc = 0.0
        
#         if verbose:
#         # if verbose and (epoch + 1) % 10 == 0:
#             print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}%")
#             # print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

#         # Step the scheduler (if provided) New !
#         if scheduler is not None:
#             scheduler.step()
#             if (epoch + 1) in [56, 57, 71, 72] and verbose:
#                 current_lr = optimizer.param_groups[0]['lr']
#                 print(f" -> Scheduler Step! Current LR: {current_lr}")
    
#     return history


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
    
    # 记录详细历史用于分析
    history = {
        'train_loss': [],
        'val_loss': [],
        'test_acc': [],
        'early_stop_epoch': None, # 哪一个 Epoch 触发的
        'early_stop_iter': None,  # [New] 哪一个 Iteration 触发的 (Figure 1 需要)
        'early_stop_test_acc': None, # [New] 触发时的 Test Acc (Figure 1 需要)
        'early_stop_val_acc': None   # [New] 触发时的 Val Acc (备用)
    }
    
    # Early Stop 追踪变量
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_triggered = False
    
    # [New] 计算每个 Epoch 有多少个 Iteration (Steps)
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
        current_val_acc = 0.0 # 需要记录 Val Acc 吗？
        current_test_acc = 0.0
        
        # 先做评估
        if val_loader:
            current_val_loss, current_val_acc = evaluate_model(model, val_loader, device)
            history['val_loss'].append(current_val_loss)
            
        if test_loader:
            _, current_test_acc = evaluate_model(model, test_loader, device)
            history['test_acc'].append(current_test_acc)
            
        model.train() # 切换回训练模式

        # # 3. 检查 Early Stop (Shadow Mode)
        # # 我们只在 val_loader 存在时才检查
        # if val_loader and not early_stop_triggered:
        #     # 检查 Loss 是否下降
        #     if current_val_loss < (best_val_loss - min_delta):
        #         best_val_loss = current_val_loss
        #         patience_counter = 0
        #     else:
        #         patience_counter += 1
            
        #     # 触发判断
        #     if patience_counter >= patience:
        #         early_stop_triggered = True
                
        #         # [Important] 记录 Figure 1 所需的关键数据
        #         history['early_stop_epoch'] = epoch + 1
        #         history['early_stop_iter'] = (epoch + 1) * steps_per_epoch # 转换为 Iterations
        #         history['early_stop_test_acc'] = current_test_acc # 记录此时的 Test Acc
        #         history['early_stop_val_acc'] = current_val_acc
                
        #         if verbose:
        #             print(f"   [Tracker] 🚩 Early Stop Condition at Iter {history['early_stop_iter']} "
        #                   f"(Test Acc: {current_test_acc:.2f}%)")
                    
        # 3. 检查 Early Stop (Shadow Mode)
        # 我们只在 val_loader 存在时才检查
        if val_loader and not early_stop_triggered:
            
            # === [关键修改 START] ===
            # 热身保护：在前 50 个 Epoch，由于 LR=0.1 导致 Loss 震荡剧烈，
            # 我们强制不计算 Patience，避免 Tracker 被噪音误导而过早触发。
            # ResNet 通常在 Epoch 56 第一次衰减 LR，真正的收敛发生在那之后。
            if epoch < 50:
                patience_counter = 0 # 强制清零，保持“满血”状态
                # 依然更新 best_val_loss，防止 decay 后的 loss 比初期还高（虽然不太可能）
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
            
            # 正常逻辑：Epoch 50 之后开始严查
            else:
                # 检查 Loss 是否下降
                if current_val_loss < (best_val_loss - min_delta):
                    best_val_loss = current_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 触发判断
                if patience_counter >= patience:
                    early_stop_triggered = True
                    
                    history['early_stop_epoch'] = epoch + 1
                    history['early_stop_iter'] = (epoch + 1) * steps_per_epoch
                    history['early_stop_test_acc'] = current_test_acc 
                    history['early_stop_val_acc'] = current_val_acc
                    
                    if verbose:
                        print(f"   [Tracker] 🚩 Early Stop Condition at Iter {history['early_stop_iter']} "
                              f"(Epoch {epoch+1}, Test Acc: {current_test_acc:.2f}%)")
             # === [关键修改 END] ===

        # 4. Logging & Scheduler
        if verbose:
             print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {current_val_loss:.4f} | Val Acc: {current_val_acc:.2f}% | Test Acc: {current_test_acc:.2f}%")

        if scheduler is not None:
            scheduler.step()
            if (epoch + 1) in [56, 57, 71, 72] and verbose:
                current_lr = optimizer.param_groups[0]['lr']
                print(f" -> Scheduler Step! Current LR: {current_lr}")
    
    # 如果跑完了所有 Epoch 还没触发 Early Stop，
    # 通常取最后一个 Epoch 的数据作为 "Early Stop" 点 (或者视为没有收敛)
    # 论文中通常指 "Iterations to reach early-stopping criteria"，如果没达到，可能取最大值。
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

        # Check if we are running the ResNet-18/-20 config
        if 'resnet' in config.get('description', '').lower():
            # Milestone 1: ~20k/30k iterations (2/3 of training)
            m1 = int(epochs_per_round * 0.66)
            # Milestone 2: ~25k/30k iterations (5/6 of training)
            m2 = int(epochs_per_round * 0.83)
            
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=[m1, m2], 
                gamma=0.1
            )
            
            if verbose:
                print(f"Scheduler enabled: MultiStepLR at epochs {m1} and {m2}")

        
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=epochs_per_round,
            mask=current_mask,
            scheduler=scheduler,  # <--- Critical Change (New!)
            test_loader=test_loader,  # <--- Critical Change (New!)
            val_loader=val_loader,  # <--- Critical Change (New!)
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
        'remaining_params': count_nonzero_parameters(model, mask)
        # 'mask': mask
    }




def one_shot_pruning_efficient(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer_class: Type[torch.optim.Optimizer],
    optimizer_kwargs: Dict,
    target_sparsity: float,  # Percentage to REMOVE (e.g., 0.8)
    device: torch.device,
    epochs: int = 86,
    trained_baseline_state: Optional[Dict] = None,
    initial_weights_state: Optional[Dict] = None,
    verbose: bool = True
) -> Dict:
    """
    LTH One-shot pruning: 1. Identify Mask -> 2. Reset -> 3. Retrain.
    """
    model = model.to(device)

    # --- Step 1: Generate Mask from Trained Baseline ---
    if trained_baseline_state is None:
        raise ValueError("trained_baseline_state is required for one-shot pruning.")
    
    model.load_state_dict(trained_baseline_state)
    
    # Global magnitude pruning
    all_weights = torch.cat([p.data.flatten() for n, p in model.named_parameters() if 'weight' in n])
    threshold = torch.quantile(torch.abs(all_weights), target_sparsity)
    
    mask = {n: (torch.abs(p.data) >= threshold).float() 
            for n, p in model.named_parameters() if 'weight' in n}

    # --- Step 2: Reset to Initial Theta_0 ---
    if initial_weights_state is None:
        raise ValueError("initial_weights_state is required to find the winning ticket.")
        
    model.load_state_dict(initial_weights_state)
    
    # Ensure apply_mask is available in your pruning.py scope
    apply_mask(model, mask)

    # --- Step 3: Retrain the Sparse Network ---
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss()
    
    # Standard ResNet scheduler; adjust milestones if using different models
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[56, 71], gamma=0.1)
    
    if verbose:
        print(f"Retraining One-Shot ticket (Remaining: {(1-target_sparsity)*100:.2f}%)...")

    # Ensure train_model is available in your pruning.py scope
    history = train_model(
        model, train_loader, optimizer, criterion, device,
        epochs=epochs, mask=mask, scheduler=scheduler,
        test_loader=test_loader, verbose=verbose
    )

    return {
        # 'remaining_pct': (1.0 - target_sparsity) * 100,
        'remaining_params_pct': (1.0 - target_sparsity) * 100,     # Match the key ax1 expects
        'test_accuracy': history['test_acc'][-1],
        'test_loss': history['train_loss'][-1], # placeholder for loss
        'remaining_params': count_nonzero_parameters(model, mask), # Use your helper
        'history': history
        # 'mask': mask
    }



def random_reinit_pruning(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer_class: type,
    optimizer_kwargs: Dict,
    device: torch.device,
    mask: Dict[str, torch.Tensor],
    val_loader: Optional[torch.utils.data.DataLoader] = None, # [新增] 保持接口一致
    epochs: int = 50,
    verbose: bool = True
) -> Dict:
    """
    Random reinitialization pruning (Control Experiment).
    Returns a dictionary structure identical to one item in the iterative_pruning results list.
    """
    model = model.to(device)
    
    # 0. Calculate total parameters (before applying mask, to get denominator)
    # We assume the model structure passed in is the full structure
    total_params = sum(p.numel() for p in model.parameters())

    # 1. Randomly reinitialize (Using Xavier/Glorot as discussed for ResNet)
    # 确保这里和你的 _weights_init 逻辑一致
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    # 2. Apply Mask (Structure)
    apply_mask(model, mask)
    
    # Calculate sparsity stats
    remaining_params = count_nonzero_parameters(model, mask)
    remaining_pct = 100.0 * remaining_params / total_params

    if verbose:
        print("\n" + "=" * 70)
        print(f"Starting Random Reinit Control Experiment")
        print(f"Sparsity Level: {remaining_pct:.2f}% weights remaining")
        print("=" * 70)
    
    # 3. Setup Optimizer & Scheduler
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss()

    scheduler = None
    if epochs > 0:
        # Match the scheduler logic from iterative_pruning exactly
        m1 = int(epochs * 0.66)
        m2 = int(epochs * 0.83)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[m1, m2], gamma=0.1
        )
        if verbose:
            print(f"Scheduler enabled: MultiStepLR at epochs {m1} and {m2}")

    # 4. Train
    # Capture the history return value!
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
    
    # 5. Evaluate Final Performance
    test_loss, test_accuracy = evaluate_model(model, test_loader, device)
    
    if verbose:
        print(f"\nRandom Reinit Result:")
        print(f"  Test Accuracy: {test_accuracy:.2f}%")
        print(f"  Remaining Params: {remaining_pct:.2f}%")
        print("=" * 70 + "\n")
    
    # 6. Return Data (Matching iterative_pruning structure)
    return {
        'test_accuracy': test_accuracy,       # 最终精度 (用于画点)
        'test_loss': test_loss,
        'remaining_params': remaining_params,
        'remaining_params_pct': remaining_pct, # 横坐标 (X-axis)
        'history': history,                   # 训练曲线 (包含每个epoch的acc, 用于画过程)
        'mask': None # Random Reinit 不需要存 mask，省点内存，如果需要可以存 mask
    }

# def random_reinit_pruning(
#     model: nn.Module,
#     train_loader: torch.utils.data.DataLoader,
#     test_loader: torch.utils.data.DataLoader,
#     optimizer_class: type,
#     optimizer_kwargs: Dict,
#     device: torch.device,
#     mask: Dict[str, torch.Tensor],
#     epochs: int = 50,
#     verbose: bool = True
# ) -> Dict:
#     """
#     Random reinitialization pruning (baseline for comparison).
    
#     Instead of resetting to original θ₀, randomly reinitialize.
#     This should perform worse than iterative pruning (winning ticket).
    
#     Args:
#         model: PyTorch model
#         train_loader: Training data loader
#         test_loader: Test data loader
#         optimizer_class: Optimizer class
#         optimizer_kwargs: Optimizer arguments
#         device: Device to train on
#         mask: Pruning mask from iterative pruning
#         epochs: Training epochs
#         verbose: Whether to print progress
    
#     Returns:
#         Dictionary with results
#     """
#     model = model.to(device)
    
#     # Randomly reinitialize
#     def init_weights(m):
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             nn.init.xavier_normal_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
    
#     model.apply(init_weights)
#     apply_mask(model, mask)
    
#     if verbose:
#         print("Random Reinitialization Pruning: Training...")
    
#     # Train
#     optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
#     criterion = nn.CrossEntropyLoss()


#     # 3. Setup Scheduler (Critical for ResNet comparison)
#     scheduler = None
#     # 简单复用 iterative_pruning 里的逻辑
#     m1 = int(epochs * 0.66)
#     m2 = int(epochs * 0.83)
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(
#         optimizer, milestones=[m1, m2], gamma=0.1
#     )

#     train_model(model, train_loader, test_loader, optimizer, criterion, scheduler=scheduler, device=device, epochs=epochs, mask=mask, verbose=verbose)
    
#     # Evaluate
#     test_loss, test_accuracy = evaluate_model(model, test_loader, device)
    
#     return {
#         'test_accuracy': test_accuracy,
#         'test_loss': test_loss,
#         'remaining_params': count_nonzero_parameters(model, mask)
#     }


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