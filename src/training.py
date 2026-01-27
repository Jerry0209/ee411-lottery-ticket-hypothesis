import copy
import torch
import torch.nn as nn
import torch.optim as optim
from .pruning import apply_mask


def validate(model, device, val_loader, criterion):
    """Evaluate model on validation set."""
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
    """Train model with early stopping based on validation loss."""
    model.train()

    results = {
        'iterations': [],
        'val_losses': [],
        'val_accs': [],
        'train_accs': [],
        'early_stop_iter': None,
        'early_stop_val_loss': float('inf'),
        'early_stop_test_acc': 0.0,
        'early_stop_val_acc': 0.0,
        'train_acc_at_early_stop': 0.0,
    }

    iteration = 0
    train_iter = iter(train_loader)
    best_val_loss = float('inf')
    best_state = None
    while iteration < max_iterations:
        try:
            data, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data, target = next(train_iter)
        
        data, target = data.to(device), target.to(device)
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if mask_dict is not None:
            apply_mask(model, mask_dict)
        
        iteration += 1
        if iteration % eval_every == 0 or iteration == max_iterations:
            val_loss, val_acc = validate(model, device, val_loader, criterion)
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
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                results['early_stop_iter'] = iteration
                results['early_stop_val_loss'] = val_loss
                results['early_stop_val_acc'] = val_acc
                results['train_acc_at_early_stop'] = train_acc
            
    results['val_acc_at_final'] = results['val_accs'][-1]
    results['train_acc_at_final'] = results['train_accs'][-1]

    if test_loader is not None:
        _, test_acc = validate(model, device, test_loader, criterion)
        results['final_test_acc'] = test_acc
    else:
        results['final_test_acc'] = results['val_accs'][-1]

    if test_loader is not None and best_state is not None:
        current_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        _, early_stop_test_acc = validate(model, device, test_loader, criterion)
        results['early_stop_test_acc'] = early_stop_test_acc
        model.load_state_dict(current_state)
    else:
        results['early_stop_test_acc'] = results['early_stop_val_acc']

    results['learning_curve'] = {
        'iterations': results['iterations'],
        'val_accs': results['val_accs'],
        'val_losses': results['val_losses'],
        'train_accs': results['train_accs']
    }

    return results


def create_optimizer(model, config):
    """Create optimizer based on config (Adam or SGD)."""
    if config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    return optimizer