from torch.optim.lr_scheduler import LRScheduler


class WarmupLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps=10000,
        last_epoch=-1,
        verbose=False,
        **kwargs
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            learning_rates = [
                base_lr * self._step_count / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            learning_rates = self.base_lrs
        return learning_rates
