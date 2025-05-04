import torch

class LambdaScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Custom scheduler for lambdas' learning rate in SA-PINN.

    :param optimizer: Wrapped optimizer.
    :param N_warmup: Number of warmup epochs.
    :param base_scheduler: Base scheduler from torch.optim.lr_scheduler.
    :param base_scheduler_kwargs: Keyword arguments for the base scheduler.
    :param last_epoch: The index of the last epoch. Default: -1.
    """
    def __init__(self, optimizer, N_warmup, base_scheduler, last_epoch=-1, **base_scheduler_kwargs):
        self.N_warmup = N_warmup
        self.base_scheduler = base_scheduler(optimizer, **base_scheduler_kwargs)
        super(LambdaScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute the learning rate for each parameter group.
        """
        if self.last_epoch < self.N_warmup:
            # During warmup, keep the learning rate constant
            return [0.0 for base_lr in self.base_lrs]
        else:
            # After warmup, use the base scheduler's learning rate
            return self.base_scheduler.get_last_lr()

    def step(self, epoch=None):
        """
        Update the learning rate based on the current epoch.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.last_epoch >= self.N_warmup:
            # Step the base scheduler after warmup
            self.base_scheduler.step(epoch - self.N_warmup)
        super(LambdaScheduler, self).step(epoch)