from .basic import AutoStep


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

        return optim.SGD(parameters, lr=self.lr_init, **kwargs)
