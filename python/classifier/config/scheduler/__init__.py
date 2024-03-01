from classifier.nn.schedule import MultiStepBS, Schedule


class FixedStep(Schedule):
    '''
    Use a much larger training batches and learning rate by default [1]_.

    .. [1] https://arxiv.org/pdf/1711.00489.pdf
    '''
    epoch = 20

    bs_init = 2**10
    bs_milestones = [1, 3, 6, 10]
    bs_scale = 2
    lr_init = 1.0e-2
    lr_milestones = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    lr_scale = 0.25

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


class AutoStep(FixedStep):
    lr_threshold = 1e-4
    lr_patience = 1
    lr_cooldown = 1
    lr_min = 2e-4

    def lr_scheduler(self, optimizer, **kwargs):
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=self.lr_scale,
            threshold=self.lr_threshold,
            patience=self.lr_patience,
            cooldown=self.lr_cooldown,
            min_lr=self.lr_min,
            **kwargs)


class FinetuneStep(AutoStep):
    lr_init = 1.0e-4
    lr_scale = 0.5
    lr_threshold = 1e-3
    lr_patience = 0
    lr_cooldown = 0
    lr_min = 1e-9


class FinetuneStepSGD(FinetuneStep):  # PLAN test performance
    def optimizer(self, parameters, **kwargs):
        import torch.optim as optim
        return optim.SGD(
            parameters,
            lr=self.lr_init,
            **kwargs
        )
