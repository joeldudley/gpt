import torch
from torch.optim import AdamW

from simplegpt.config.config import WEIGHT_DECAY


def get_adamw_optimizer(named_modules, named_params):
    decay_params, no_decay_params = _get_params_groups(named_modules, named_params)
    optimiser_groups = [{"params": decay_params, "weight_decay": WEIGHT_DECAY},
                        {"params": no_decay_params, "weight_decay": 0.0}]
    return AdamW(optimiser_groups)


def _get_params_groups(named_modules, named_params):
    all_params = {param for _, param in named_params}
    params_to_decay = {param for _, module in named_modules for param_name, param in module.named_parameters()
                       if param_name.endswith('weight') and isinstance(module, torch.nn.Linear)}
    return list(params_to_decay), list(all_params - params_to_decay)
