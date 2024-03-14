from classifier.nn.dataset import io_loader
from classifier.nn.schedule import Schedule
from classifier.utils import noop


class _SkimBS(noop):
    def __init__(self, dataset):
        self.dataloader = io_loader(dataset, shuffle=False, drop_last=False)


class SkimStep(Schedule):
    epoch: int = 1

    def optimizer(cls, _, **__):
        return noop()

    def lr_scheduler(self, _, **__):
        return noop()

    def bs_scheduler(self, dataset, **__):
        return _SkimBS(dataset)
