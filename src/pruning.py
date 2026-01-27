import torch


def create_random_mask(model, sparsity):
    mask_dict = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask = torch.ones_like(param)
            num_params = param.numel()
            num_to_prune = int(num_params * (1 - sparsity))
            flat_mask = mask.view(-1)
            prune_indices = torch.randperm(num_params)[:num_to_prune]
            flat_mask[prune_indices] = 0
            
            mask_dict[name] = mask.view_as(param)
        else:
            mask_dict[name] = torch.ones_like(param)
    
    return mask_dict


def create_magnitude_mask_layerwise(model, target_sparsity, existing_mask=None,
                                    layer_specific_rates=False, prune_rate=0.2):
    mask_dict = {}
    
    for name, param in model.named_parameters():
        if 'weight' not in name:
            mask_dict[name] = torch.ones_like(param)
            continue
        
        weights = param.data.abs()
        if existing_mask is not None:
            weights = weights * existing_mask[name].to(weights.device)
        
        if layer_specific_rates and 'fc3' in name:
            layer_prune_rate = prune_rate / 2
        else:
            layer_prune_rate = prune_rate
        
        num_weights = param.numel()
        if existing_mask is None:
            num_to_keep = int(num_weights * target_sparsity)
        else:
            current_active = (existing_mask[name] != 0).sum().item()
            num_to_keep = int(current_active * (1 - layer_prune_rate))
        
        if num_to_keep == 0:
            mask = torch.zeros_like(param)
        elif num_to_keep >= num_weights:
            mask = torch.ones_like(param)
        else:
            flat_weights = weights.view(-1)
            threshold = torch.topk(flat_weights, num_to_keep)[0][-1]
            mask = (weights >= threshold).float()
        
        mask_dict[name] = mask
    
    return mask_dict


def apply_mask(model, mask_dict):
    for name, param in model.named_parameters():
        if name in mask_dict:
            param.data *= mask_dict[name].to(param.device)


def combine_masks(mask1, mask2):
    combined = {}
    for name in mask1:
        combined[name] = mask1[name] * mask2[name]
    return combined


def count_parameters(mask_dict):
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
    if mask_dict is None:
        return 1.0
    
    total_params = sum(p.numel() for name, p in model.named_parameters() if 'weight' in name)
    remaining_params = count_parameters(mask_dict)[0]
    
    return remaining_params / total_params if total_params > 0 else 0.0