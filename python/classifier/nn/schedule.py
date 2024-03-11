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

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @abstractmethod
    def optimizer(cls, parameters, **kwargs) -> optim.Optimizer: ...

    @abstractmethod
    def bs_scheduler(cls, dataset: Dataset, **kwargs) -> BSScheduler: ...

    @abstractmethod
    def lr_scheduler(cls, optimizer: optim.Optimizer, **kwargs) -> LRScheduler: ...


class MultiStepBS(BSScheduler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        milestones: Optional[Iterable[int]] = None,
        gamma: Optional[float] = None,
        **kwargs,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.milestones = sorted(milestones or [])
        self.gamma = gamma
        self.kwargs = kwargs

        self._epoch = 0
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
        if epoch is None:
            self._epoch += 1
        else:
            self._epoch = epoch
        milestone = bisect_right(self.milestones, self._epoch)
        self._bs = int(self.batch_size * (self.gamma**milestone))
