from ...utils import noop
from ..dataset import io_loader
from . import Schedule


class _SkimBS(noop):
    def __init__(self, dataset):
        self.dataloader = io_loader(dataset, shuffle=False, drop_last=False)


class Skim(Schedule):
    epoch: int = 1

    def optimizer(cls, _, **__):
        return noop()

    def lr_scheduler(self, _, **__):
        return noop()

    def bs_scheduler(self, dataset, **__):
        return _SkimBS(dataset)
