from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_right
from typing import TYPE_CHECKING, Iterable, Optional

from .dataset import mp_loader

if TYPE_CHECKING:
    from torch import optim
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader, Dataset


class BSScheduler(ABC):
    dataloader: DataLoader

    @abstractmethod
    def step(self, epoch: int = None): ...


class Schedule(ABC):
    epoch: int

    @abstractmethod
    def optimizer(cls, parameters, **kwargs) -> optim.Optimizer: ...

    @abstractmethod
    def bs_scheduler(cls, dataset: Dataset, **kwargs) -> BSScheduler: ...

    @abstractmethod
    def lr_scheduler(cls, optimizer: optim.Optimizer, **kwargs) -> LRScheduler: ...


class MilestoneStep:
    def __init__(self, milestones: Optional[Iterable[int]] = None):
        self.reset()
        self._milestones = []
        self.milestones = milestones

    @property
    def milestone(self):
        return self._milestone

    @property
    def milestones(self):
        return self._milestones

    @milestones.setter
    def milestones(self, milestones: Iterable[int] = None):
        self._milestones = sorted(milestones or [])

    def reset(self):
        self._step = 0
        self._milestone = 0

    def step(self, step: int = None):
        if step is None:
            self._step += 1
        else:
            self._step = step
        milestone = bisect_right(self.milestones, self._step)
        changed = milestone != self.milestone
        self._milestone = milestone
        return changed


class MultiStepBS(MilestoneStep, BSScheduler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        milestones: Optional[Iterable[int]] = None,
        gamma: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(milestones=milestones)
        self.dataset = dataset
        self.batch_size = batch_size
        self.gamma = gamma
        self.kwargs = kwargs

        self._bs = batch_size
        self._dataloader: DataLoader = None

    @property
    def dataloader(self):
        if self._dataloader is None or self._dataloader.batch_size != self._bs:
            self._dataloader = mp_loader(
                self.dataset, batch_size=self._bs, **self.kwargs
            )
        return self._dataloader

    def step(self, epoch: int = None):
        super(MultiStepBS, self).step(epoch)
        self._bs = int(self.batch_size * (self.gamma**self.milestone))
