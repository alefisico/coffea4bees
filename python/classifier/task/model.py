from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .special import interface
from .task import Task

if TYPE_CHECKING:
    from torch.utils.data import StackDataset

    from ..process.device import Device


class Model(Task):
    @interface
    def train(self) -> list[ModelTrainer]:
        """
        Pepare models for training.
        """
        ...

    @interface
    def evaluate(self):  # TODO evaluation
        ...


class ModelTrainer(Protocol):
    def __call__(self, device: Device, datasets: StackDataset) -> dict[str]: ...


class ModelRunner(Protocol):  # TODO evaluation
    ...
