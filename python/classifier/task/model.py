from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .task import Task, interface

if TYPE_CHECKING:
    from torch.utils.data import StackDataset

    from ..process.device import Device


class Model(Task):
    @interface
    def train(self, datasets: StackDataset) -> list[ModelTrainer]:
        """
        Preprocess datasets and prepare models for training.
        """
        ...

    @interface
    def evaluate(self):  # TODO evaluation
        ...


class ModelTrainer(Protocol):
    def __call__(self, device: Device) -> dict[str]:
        ...


class ModelRunner(Protocol):  # TODO evaluation
    ...
