from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..process.state import Cascade
from . import task

if TYPE_CHECKING:
    from torch.utils.data import Dataset as TorchDataset


class Dataset(task.Task):
    @task.interface
    def train(self) -> list[TrainingSetLoader]:
        """
        Prepare training set loaders.
        """
        ...

    @task.interface
    def evaluate(self):  # TODO evaluation
        ...


class TrainingSetLoader(Protocol):
    def __call__(self) -> dict[str, TorchDataset]:
        """
        Load training set.
        """
        ...


class EvaluationSetLoader(Protocol):  # TODO evaluation
    ...


class Setting(Cascade):
    io_step: int = 1_000_000
    dataloader_shuffle: bool = True
