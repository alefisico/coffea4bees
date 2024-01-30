import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from ..network import MultiStepBS, Schedule


class FixedStep(Schedule):
    '''
    https://arxiv.org/pdf/1711.00489.pdf much larger training batches and learning rate inspired by this paper
    '''
    num_workers = 0
    epoch = 20

    bs_eval = 2**15
    bs_train = 2**10
    bs_milestones = [1, 3, 6, 10]
    bs_scale = 2
    lr_init = 1.0e-2
    lr_milestones = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    lr_scale = 0.25

    @classmethod
    def optimizer(cls, parameters, **kwargs):
        return optim.Adam(
            parameters,
            lr=cls.lr_init,
            # amsgrad=True, # PLAN test performance
            **kwargs
        )

    @classmethod
    def bs_scheduler(cls, columns, dataset, **kwargs):
        return MultiStepBS(
            columns=columns,
            dataset=dataset,
            batch_size=cls.bs_train,
            milestones=cls.bs_milestones,
            gamma=cls.bs_scale,
            **kwargs
        )

    @classmethod
    def lr_scheduler(cls, optimizer, **kwargs):
        return MultiStepLR(
            optimizer=optimizer,
            milestones=cls.lr_milestones,
            gamma=cls.lr_scale,
            **kwargs
        )


class AutoStep(FixedStep):
    lr_threshold = 1e-4
    lr_patience = 1
    lr_cooldown = 1
    lr_min = 2e-4

    @classmethod
    def lr_scheduler(cls, optimizer, **kwargs):
        return ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=cls.lr_scale,
            threshold=cls.lr_threshold,
            patience=cls.lr_patience,
            cooldown=cls.lr_cooldown,
            min_lr=cls.lr_min,
            **kwargs)


class FinetuneStep(AutoStep):
    lr_init = 1.0e-4
    lr_scale = 0.5
    lr_threshold = 1e-3
    lr_patience = 0
    lr_cooldown = 0
    lr_min = 1e-9


class FinetuneStepSGD(FinetuneStep):  # PLAN test performance
    @classmethod
    def optimizer(cls, parameters, **kwargs):
        return optim.SGD(
            parameters,
            lr=cls.lr_init,
            **kwargs
        )
