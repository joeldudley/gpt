import torch
from torch.optim import AdamW

from gpt.constants import WEIGHT_DECAY


def get_adamw_optimizer(named_modules, named_params):
    decay_params, no_decay_params = _get_params_by_type(named_modules, named_params)
    optimiser_groups = [{"params": decay_params, "weight_decay": WEIGHT_DECAY},
                        {"params": no_decay_params, "weight_decay": 0.0}]
    return AdamW(optimiser_groups)


def _get_params_by_type(named_modules, named_params):
    params_to_decay = _get_linear_weight_params(named_modules)
    param_dict = {param_name: param for param_name, param in named_params}
    decay_params = [param_dict[param_name] for param_name in params_to_decay]
    no_decay_params = [param_dict[param_name] for param_name in param_dict.keys() - params_to_decay]
    return decay_params, no_decay_params


def _get_linear_weight_params(named_modules):
    return {(module_name + '.' if module_name else '') + param_name
            for module_name, module in named_modules
            for param_name, _ in module.named_parameters()
            if param_name.endswith('weight') and isinstance(module, torch.nn.Linear)}
