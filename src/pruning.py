"""
Pruning utilities for lottery ticket experiments.
"""

import torch


def create_random_mask(model, sparsity):
    """
    Create a random binary mask for the model.
    
    Args:
        model: PyTorch model
        sparsity: Fraction of weights to KEEP (0.5 = keep 50%, prune 50%)
    
    Returns:
        mask_dict: Dictionary mapping parameter names to binary masks
    """
    mask_dict = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Create mask of all 1s
            mask = torch.ones_like(param)
            num_params = param.numel()
            num_to_prune = int(num_params * (1 - sparsity))
            
            # Randomly select indices to prune
            flat_mask = mask.view(-1)
            prune_indices = torch.randperm(num_params)[:num_to_prune]
            flat_mask[prune_indices] = 0
            
            mask_dict[name] = mask.view_as(param)
        else:
            # Don't prune biases
            mask_dict[name] = torch.ones_like(param)
    
    return mask_dict


def create_magnitude_mask_layerwise(model, target_sparsity, existing_mask=None, 
                                     layer_specific_rates=False, prune_rate=0.2):
    """
    Create magnitude-based mask with LAYER-WISE pruning.
    
    For iterative pruning: prunes prune_rate% of REMAINING weights in each layer.
    For one-shot pruning: prunes to reach target_sparsity directly.
    
    Args:
        model: PyTorch model
        target_sparsity: Target fraction of weights to keep
        existing_mask: Cumulative mask from previous rounds (for iterative)
        layer_specific_rates: If True, prune output layer at half rate
        prune_rate: Fraction to prune per round (for iterative)
    
    Returns:
        mask_dict: Dictionary of binary masks
    """
    mask_dict = {}
    
    for name, param in model.named_parameters():
        if 'weight' not in name:
            # Don't prune biases
            mask_dict[name] = torch.ones_like(param)
            continue
        
        # Get weight magnitudes
        weights = param.data.abs()
        
        # If existing mask, only consider surviving weights
        if existing_mask is not None:
            weights = weights * existing_mask[name].to(weights.device)
        
        # Determine pruning rate for this layer
        if layer_specific_rates and 'fc3' in name:
            # Output layer: prune at half rate
            layer_prune_rate = prune_rate / 2
        else:
            layer_prune_rate = prune_rate
        
        # Calculate number of weights to keep
        num_weights = param.numel()
        
        if existing_mask is None:
            # One-shot: prune directly to target sparsity
            num_to_keep = int(num_weights * target_sparsity)
        else:
            # Iterative: prune prune_rate% of remaining weights
            current_active = (existing_mask[name] != 0).sum().item()
            num_to_keep = int(current_active * (1 - layer_prune_rate))
        
        # Create mask
        if num_to_keep == 0:
            mask = torch.zeros_like(param)
        elif num_to_keep >= num_weights:
            mask = torch.ones_like(param)
        else:
            # Find threshold: smallest magnitude among top k
            flat_weights = weights.view(-1)
            threshold = torch.topk(flat_weights, num_to_keep)[0][-1]
            
            # Keep weights with magnitude >= threshold
            mask = (weights >= threshold).float()
        
        mask_dict[name] = mask
    
    return mask_dict


def apply_mask(model, mask_dict):
    """
    Apply binary mask to model parameters (sets pruned weights to 0).
    
    Args:
        model: PyTorch model
        mask_dict: Dictionary of binary masks
    """
    for name, param in model.named_parameters():
        if name in mask_dict:
            param.data *= mask_dict[name].to(param.device)


def combine_masks(mask1, mask2):
    """
    Combine two masks (AND operation).
    A weight survives only if it survives in BOTH masks.
    
    Args:
        mask1, mask2: Dictionaries of binary masks
    
    Returns:
        combined_mask: Dictionary of combined masks
    """
    combined = {}
    for name in mask1:
        combined[name] = mask1[name] * mask2[name]
    return combined


def count_parameters(mask_dict):
    """
    Count number of non-zero parameters in mask.
    
    Returns:
        total: Total number of active parameters
        per_layer: Dict of active parameters per layer
    """
    # Handle None case
    if mask_dict is None:
        return 0, {}
    
    total = 0
    per_layer = {}
    
    for name, mask in mask_dict.items():
        if 'weight' in name:
            count = (mask != 0).sum().item()
            per_layer[name] = count
            total += count
    
    return total, per_layer


def get_sparsity(mask_dict, model):
    """
    Calculate actual sparsity (fraction of weights remaining).
    
    Returns:
        sparsity: Fraction of weights remaining (0 to 1)
    """
    # Handle case where mask_dict is None (no pruning)
    if mask_dict is None:
        return 1.0
    
    total_params = sum(p.numel() for name, p in model.named_parameters() if 'weight' in name)
    remaining_params = count_parameters(mask_dict)[0]
    
    return remaining_params / total_params if total_params > 0 else 0.0