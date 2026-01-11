"""
Training utilities for lottery ticket experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from .pruning import apply_mask


def validate(model, device, val_loader, criterion):
    """
    Validate model on validation set.
    
    Returns:
        val_loss: Average validation loss
        val_acc: Validation accuracy (%)
    """
    model.eval()
    val_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / len(val_loader.dataset)
    
    return val_loss, val_acc


def train_with_early_stop_tracking(model, device, train_loader, val_loader,
                                   optimizer, criterion, max_iterations=50000,
                                   mask_dict=None, eval_every=100, verbose=False,
                                   test_loader=None):
    """
    Train model and track validation loss for early stopping detection.

    This function trains for a FIXED number of iterations and retroactively
    determines where early stopping would have occurred (minimum validation loss).

    Args:
        model: PyTorch model
        device: Device to train on
        train_loader: Training data loader
        val_loader: Validation data loader (used for early stopping)
        optimizer: Optimizer
        criterion: Loss function
        max_iterations: Maximum training iterations
        mask_dict: Binary mask to enforce sparsity (optional)
        eval_every: Evaluate every N iterations
        verbose: Print progress
        test_loader: Optional test data loader for final evaluation

    Returns:
        results: Dictionary containing:
            - early_stop_iter: Iteration of minimum validation loss
            - early_stop_val_loss: Minimum validation loss
            - early_stop_test_acc: Validation accuracy at early stop (used for comparison)
            - train_acc_at_early_stop: Training accuracy at early stop
            - final_test_acc: True test accuracy at end (if test_loader provided)
            - learning_curve: Dict with iterations, val_accs, val_losses
    """
    model.train()
    
    results = {
        'iterations': [],
        'val_losses': [],
        'val_accs': [],
        'train_accs': [],
        'early_stop_iter': None,
        'early_stop_val_loss': float('inf'),
        'early_stop_test_acc': 0.0,
        'train_acc_at_early_stop': 0.0,
    }
    
    iteration = 0
    train_iter = iter(train_loader)
    best_val_loss = float('inf')
    
    while iteration < max_iterations:
        # Get next batch
        try:
            data, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data, target = next(train_iter)
        
        data, target = data.to(device), target.to(device)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Re-apply mask (enforce sparsity)
        if mask_dict is not None:
            apply_mask(model, mask_dict)
        
        iteration += 1
        
        # Periodic evaluation
        if iteration % eval_every == 0 or iteration == max_iterations:
            val_loss, val_acc = validate(model, device, val_loader, criterion)
            
            # Calculate training accuracy on current batch (approximate)
            model.eval()
            with torch.no_grad():
                train_output = model(data)
                train_pred = train_output.argmax(dim=1, keepdim=True)
                train_correct = train_pred.eq(target.view_as(train_pred)).sum().item()
                train_acc = 100. * train_correct / len(target)
            
            results['iterations'].append(iteration)
            results['val_losses'].append(val_loss)
            results['val_accs'].append(val_acc)
            results['train_accs'].append(train_acc)
            
            # Track best validation loss (early stopping point)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                results['early_stop_iter'] = iteration
                results['early_stop_val_loss'] = val_loss
                results['early_stop_test_acc'] = val_acc
                results['train_acc_at_early_stop'] = train_acc
            
            if verbose and iteration % 1000 == 0:
                print(f"  Iter {iteration}/{max_iterations} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                      f"Best at iter {results['early_stop_iter']}")
    
    # Store final iteration metrics
    results['val_acc_at_final'] = results['val_accs'][-1]
    results['train_acc_at_final'] = results['train_accs'][-1]

    # Evaluate on test set if provided (true test accuracy)
    if test_loader is not None:
        _, test_acc = validate(model, device, test_loader, criterion)
        results['final_test_acc'] = test_acc
    else:
        # Fallback: use validation accuracy as proxy
        results['final_test_acc'] = results['val_accs'][-1]

    # Create learning curve dict
    results['learning_curve'] = {
        'iterations': results['iterations'],
        'val_accs': results['val_accs'],  # Renamed from test_accs for clarity
        'val_losses': results['val_losses'],
        'train_accs': results['train_accs']
    }

    return results


def create_optimizer(model, config):
    """
    Create optimizer based on config.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
    
    Returns:
        optimizer: Configured optimizer
    """
    if config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    return optimizer