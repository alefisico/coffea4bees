from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Iterable

import torch
import torch.types as tt
from torch import Tensor, nn
from torch.utils.data import Dataset

from ..config.setting import torch as cfg
from ..monitor.progress import MessageType, Progress
from ..monitor.usage import Usage
from ..nn.dataset import simple_loader
from ..nn.schedule import Schedule
from ..process.device import Device
from ..task.special import interface
from ..typetools import WithUUID, filename

if TYPE_CHECKING:
    from base_class.system.eos import PathLike

BatchType = dict[str, Tensor]


@dataclass(kw_only=True)
class Stage(ABC):
    name: str

    @abstractmethod
    def run(self, trainer: MultiStageTraining) -> dict[str]: ...


@dataclass(kw_only=True)
class BenchmarkStage(Stage):
    model: Model
    validation: dict[str, Dataset]

    @staticmethod
    def _loader_benchmark(dataset: Dataset, msg: MessageType = None):
        return simple_loader(
            dataset,
            report_progress=msg,
            batch_size=cfg.DataLoader.batch_eval,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def _init_benchmark(self):
        return {
            k: self._loader_benchmark(v, ("batch", "benchmark", k))
            for k, v in self.validation.items()
        }

    def _iter_benchmark(
        self, trainer: MultiStageTraining, loaders: dict[str, Iterable[BatchType]]
    ):
        benchmark = {}
        self.model.nn.eval()
        with torch.no_grad():
            for k, v in loaders.items():
                trainer.cleanup()
                Usage.checkpoint(self.name, "benchmark", k, "start")
                benchmark[k] = self.model.validate(v)
                Usage.checkpoint(self.name, "benchmark", k, "finish")
        return benchmark

    def run(self, trainer: MultiStageTraining):
        return {
            "name": self.name,
            "benchmarks": self._iter_benchmark(trainer, self._init_benchmark()),
        }


@dataclass(kw_only=True)
class TrainingStage(BenchmarkStage):
    schedule: Schedule
    training: Dataset
    validation: dict[str, Dataset] = None

    def run(self, trainer: MultiStageTraining):
        history = {
            "name": self.name,
            "parameters": self.model.n_parameters,
        }
        opt = self.schedule.optimizer(self.model.parameters())
        lr = self.schedule.lr_scheduler(opt)
        bs = self.schedule.bs_scheduler(
            self.training,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        if (
            (not cfg.Training.disable_benchmark)
            and (self.validation is not None)
            and (self.model.validate is not NotImplemented)
        ):
            benchmark = []
            validation = self._init_benchmark()
        else:
            benchmark = None
        # training
        p_epoch = Progress.new(self.schedule.epoch, ("epoch", self.name))
        logging.info(f"Start {self.name}")
        start = datetime.now()
        for epoch in range(self.schedule.epoch):
            epoch = epoch + 1
            trainer.cleanup()
            Usage.checkpoint(self.name, f"epoch{epoch}", "optimize")
            self.model.nn.train()
            if len(bs.dataloader) > 0:
                p_batch = Progress.new(len(bs.dataloader), "batch")
            for i, batch in enumerate(bs.dataloader):
                opt.zero_grad()
                loss = self.model.train(batch)
                loss.backward()
                opt.step()
                p_batch.update(i + 1, ("batch", "training", f"loss={loss.item():.4g}"))
            if benchmark is not None:
                benchmark.append(
                    {
                        "hyperparameters": {
                            "epoch": epoch,
                            "learning rate": lr.get_last_lr(),
                            "batch size": bs.dataloader.batch_size,
                        }
                        | self.model.hyperparameters,
                        "benchmarks": self._iter_benchmark(trainer, validation),
                    }
                )
            lr.step()
            bs.step()
            if self.model.step is not NotImplemented:
                self.model.step()
            p_epoch.update(epoch)
            Usage.checkpoint(self.name, f"epoch{epoch}", "finish")
        logging.info(
            f"{self.name}: run {self.schedule.epoch} epochs in {datetime.now() - start}"
        )
        if benchmark is not None:
            history["training"] = benchmark
        return history


class EvaluationStage(Stage):  # TODO evaluation
    def run(self):
        pass


@dataclass(kw_only=True)
class OutputStage(Stage):
    path: PathLike

    def run(self, _):
        return {
            "name": self.name,
            "path": str(self.path),
        }


class Model(ABC):
    def parameters(self):
        return self.nn.parameters()

    def to(self, device: tt.Device):
        self.nn.to(device)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.nn.parameters())

    @property
    def hyperparameters(self) -> dict[str]:
        return {}

    @property
    @abstractmethod
    def nn(self) -> nn.Module: ...

    @interface
    def train(self, batch: BatchType) -> Tensor: ...

    @interface(optional=True)
    def validate(self, batches: Iterable[BatchType]) -> dict: ...

    @interface  # TODO evaluation
    def evaluate(self, batch: BatchType) -> BatchType: ...

    @interface(optional=True)
    def step(self, epoch: int = None): ...


class MultiStageTraining(WithUUID, ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.metadata = kwargs
        self.name = filename(kwargs)
        self.device: tt.Device = None
        self.dataset: Dataset = None

    def cleanup(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    @cached_property
    def min_memory(self):
        return 0

    @abstractmethod
    def stages(self) -> Iterable[Stage]: ...

    def train(self, dataset: Dataset, device: Device):
        self.device = device.get(self.min_memory)
        self.dataset = dataset
        result = {
            "uuid": str(self.uuid),
            "name": self.name,
            "metadata": self.metadata,
            "history": [],
        }
        history: list[dict] = result["history"]
        for stage in self.stages():
            Usage.checkpoint("stage", stage.name, "start")
            history.append(stage.run(self))
            Usage.checkpoint("stage", stage.name, "finish")
        return result
