import torch
import torch.nn as nn
from torch.optim import Optimizer

def get_optim(model: nn.Module, optim_name: str, lr: float, weight_decay: float = 0.0) -> Optimizer:
    """
    Get an optimizer for the model.

    Args:
        model (nn.Module): The model to optimize.
        optim_name (str): The name of the optimizer.
        lr (float): The learning rate.
        weight_decay (float): The weight decay.

    Returns:
        Optimizer: The optimizer for the model.
    """
    if optim_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optim_name} not supported.")
