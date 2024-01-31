
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from torch import optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import TensorDataset

from ..dataset import NamedDataLoader


class Schedule(ABC):
    num_workers: int
    epoch: int

    bs_eval: int

    @classmethod
    @abstractmethod
    def optimizer(
            cls,
            parameters: optim.optimizer.params_t,
            **kwargs) -> optim.Optimizer:
        ...

    @classmethod
    @abstractmethod
    def bs_scheduler(
            cls,
            columns: list[str],
            dataset: TensorDataset,
            **kwargs) -> MultiStepBS:
        ...

    @classmethod
    @abstractmethod
    def lr_scheduler(
            cls,
            optimizer: optim.Optimizer,
            **kwargs) -> LRScheduler:
        ...


class MultiStepBS:
    def __init__(
            self,
            columns: list[str],
            dataset: TensorDataset,
            batch_size: int,
            milestones: Optional[list[int]] = None,
            gamma: Optional[float] = None,
            **kwargs):
        self.columns = columns
        self.dataset = dataset
        self.batch_size = batch_size
        self.milestones = milestones
        self.gamma = gamma
        self.kwargs = kwargs

        self._epoch = 0
        self._milestone = 0
        self._bs = batch_size

        self.dataloader = self._new_dataloader()

    def _new_dataloader(self):
        return NamedDataLoader(
            columns=self.columns,
            dataset=self.dataset,
            batch_size=self._bs,
            **self.kwargs)

    def step(self):
        self._epoch += 1
        if (self.milestones is not None
                and self._milestone < len(self.milestones)
                and self._epoch == self.milestones[self._milestone]):
            self._milestone += 1
            self._bs = int(self._bs * self.gamma)
            self.dataloader = self._new_dataloader()
