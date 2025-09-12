from diffusers.optimization import (Union, SchedulerType, Optional, Optimizer,
                                    TYPE_TO_SCHEDULER_FUNCTION)


def get_scheduler(name: Union[str, SchedulerType],
                  optimizer: Optimizer,
                  num_warmup_steps: Optional[int] = None,
                  num_training_steps: Optional[int] = None,
                  **kwargs):
    """
    Added kwargs vs diffuser's original implementation

    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(
            f"{name} requires `num_warmup_steps`, please provide that argument."
        )

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer,
                             num_warmup_steps=num_warmup_steps,
                             **kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(
            f"{name} requires `num_training_steps`, please provide that argument."
        )

    return schedule_func(optimizer,
                         num_warmup_steps=num_warmup_steps,
                         num_training_steps=num_training_steps,
                         **kwargs)


import torch
from torch.optim.lr_scheduler import _LRScheduler
import pytorch_lightning as pl
from diffusers.optimization import TYPE_TO_SCHEDULER_FUNCTION, SchedulerType


class SelectiveLRScheduler(_LRScheduler):

    def __init__(self,
                 optimizer,
                 base_scheduler,
                 group_indices,
                 default_lr=[1e-5, 1e-4],
                 last_epoch=-1):
        self.base_scheduler = base_scheduler
        self.group_indices = group_indices  # Indices of parameter groups to update
        self.default_lr = default_lr
        super().__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        self.base_scheduler.step()
        base_lrs = self.base_scheduler.get_last_lr()

        for idx, group in enumerate(self.optimizer.param_groups):
            if idx in self.group_indices:
                group['lr'] = base_lrs[idx]
            else:
                # Reset the learning rate to its initial value
                group['lr'] = self.default_lr[idx]
