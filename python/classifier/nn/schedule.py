from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from torch import optim
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader, Dataset


class BSScheduler(ABC):
    dataloader: DataLoader

    @abstractmethod
    def step(self):
        ...


class Schedule(ABC):
    epoch: int

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @abstractmethod
    def optimizer(
        cls,
        parameters,
        **kwargs
    ) -> optim.Optimizer:
        ...

    @abstractmethod
    def bs_scheduler(
        cls,
        dataset: Dataset,
        **kwargs
    ) -> BSScheduler:
        ...

    @abstractmethod
    def lr_scheduler(
        cls,
        optimizer: optim.Optimizer,
        **kwargs
    ) -> LRScheduler:
        ...


class MultiStepBS(BSScheduler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        milestones: Optional[Iterable[int]] = None,
        gamma: Optional[float] = None,
        **kwargs
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.milestones = sorted(milestones or [])
        self.gamma = gamma
        self.kwargs = kwargs

        self._epoch = 0
        self._milestone = 0
        self._bs = batch_size

        self.dataloader = self._new_dataloader()

    def _new_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            dataset=self.dataset,
            batch_size=self._bs,
            **self.kwargs)

    def step(self):
        self._epoch += 1
        if (self._milestone < len(self.milestones)
                and self._epoch == self.milestones[self._milestone]):
            self._milestone += 1
            self._bs = int(self.batch_size * (self.gamma ** self._milestone))
            self.dataloader = self._new_dataloader()
