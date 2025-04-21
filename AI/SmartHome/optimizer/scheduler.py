import torch
import torch.nn as nn
from torch.optim import lr_scheduler

def get_scheduler(optimizer, scheduler_name: str, **kwargs):
    """
    Get a learning rate scheduler.

    Args:
        optimizer (Optimizer): The optimizer to schedule.
        scheduler_name (str): The name of the scheduler.
        **kwargs: Additional arguments for the scheduler.

    Returns:
        lr_scheduler: The learning rate scheduler.
    """
    if scheduler_name == 'step':
        return lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name == 'multi_step':
        return lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif scheduler_name == 'exponential':
        return lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_name == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name == 'reduce_on_plateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported.")