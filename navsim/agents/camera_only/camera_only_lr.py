from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class CosineAnnealingWithDecay(CosineAnnealingWarmRestarts):
    """
    Cosine Annealing Warm Restarts with decaying max LR after each restart.
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, decay_factor=0.5, last_epoch=-1):
        """
        :param optimizer: optimizer instance
        :param T_0: number of epochs for the first cycle
        :param T_mult: cycle length multiplier
        :param eta_min: minimum learning rate
        :param decay_factor: factor to decay max LR at each restart (e.g., 0.5)
        """
        self.decay_factor = decay_factor
        self.initial_lrs = [group["lr"] for group in optimizer.param_groups]  # <── moved BEFORE super()
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)

    def step(self, epoch=None):
        """Override step() to apply LR decay at each restart."""
        super().step(epoch)
        # Apply decay only when restarting
        if self.T_cur == 0:
            for i, group in enumerate(self.optimizer.param_groups):
                self.initial_lrs[i] *= self.decay_factor
