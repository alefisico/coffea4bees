from dataclasses import dataclass

from classifier.nn.schedule import MultiStepBS, Schedule


@dataclass
class FixedStep(Schedule):
    """
    Use a much larger training batches and learning rate by default [1]_.

    .. [1] https://arxiv.org/pdf/1711.00489.pdf
    """

    epoch: int = 20

    bs_init: int = 2**10
    bs_scale: float = 2.0
    bs_milestones: list[int] = (1, 3, 6, 10)
    lr_init: float = 1.0e-2
    lr_scale: float = 0.25
    lr_milestones: list[int] = (15, 16, 17, 18, 19, 20, 21, 22, 23, 24)

    def optimizer(self, parameters, **kwargs):
        import torch.optim as optim

        return optim.Adam(
            parameters,
            lr=self.lr_init,
            # amsgrad=True, # PLAN test performance
            **kwargs
        )

    def bs_scheduler(self, dataset, **kwargs):
        return MultiStepBS(
            dataset=dataset,
            batch_size=self.bs_init,
            milestones=self.bs_milestones,
            gamma=self.bs_scale,
            **kwargs
        )

    def lr_scheduler(self, optimizer, **kwargs):
        from torch.optim.lr_scheduler import MultiStepLR

        return MultiStepLR(
            optimizer=optimizer,
            milestones=self.lr_milestones,
            gamma=self.lr_scale,
            **kwargs
        )


@dataclass
class AutoStep(FixedStep):
    lr_threshold: float = 1e-4
    lr_patience: int = 1
    lr_cooldown: int = 1
    lr_min: float = 2e-4

    def lr_scheduler(self, optimizer, **kwargs):
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        return ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=self.lr_scale,
            threshold=self.lr_threshold,
            patience=self.lr_patience,
            cooldown=self.lr_cooldown,
            min_lr=self.lr_min,
            **kwargs
        )
