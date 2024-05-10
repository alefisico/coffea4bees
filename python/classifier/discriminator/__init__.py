from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from typing import Iterable

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


@dataclass
class TrainingStage:
    name: str
    model: Model
    schedule: Schedule

    training: Dataset
    validation: Dataset = None

    do_benchmark: bool = True


class Model(ABC):
    def parameters(self):
        return self.module.parameters()

    def to(self, device: tt.Device):
        self.module.to(device)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    @property
    @abstractmethod
    def module(self) -> nn.Module: ...

    @interface
    def train(self, batch: dict[str, Tensor]) -> Tensor: ...

    @interface(optional=True)
    def validate(self, batches: Iterable[dict[str, Tensor]]) -> dict: ...

    @interface  # TODO evaluation
    def evaluate(self, batch: dict[str, Tensor]) -> dict[str, Tensor]: ...

    @interface(optional=True)
    def step(self, epoch: int = None): ...


class Classifier(WithUUID, ABC):
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
    def training_stages(
        self,
    ) -> Iterable[TrainingStage]: ...

    def train(
        self,
        dataset: Dataset,
        device: Device,
    ):
        self.device = device.get(self.min_memory)
        self.dataset = dataset
        result = {
            "uuid": str(self.uuid),
            "name": self.name,
            "metadata": self.metadata,
            "history": [],
        }
        Usage.checkpoint("classifier", "init")
        for stage in self.training_stages():
            Usage.checkpoint("classifier", stage.name, "start")
            history = {
                "name": stage.name,
                "parameters": stage.model.n_parameters,
            }
            benchmark = self._train(stage=stage)
            if benchmark:
                history["benchmark"] = benchmark
            result["history"].append(history)
            Usage.checkpoint("classifier", stage.name, "finish")
        return result

    def evaluate(
        self,
    ): ...  # TODO evaluataion

    def _train(
        self,
        stage: TrainingStage,
    ):
        model = stage.model
        schedule = stage.schedule
        # training preparation
        optimizer = schedule.optimizer(model.parameters())
        lr = schedule.lr_scheduler(optimizer)
        bs = schedule.bs_scheduler(
            stage.training,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        if (
            (not cfg.Training.disable_benchmark)
            and stage.do_benchmark
            and (model.validate is not NotImplemented)
        ):
            benchmark = []
            validation_set = {
                "training": self._validation_loader(
                    stage.training, ("batch", "validation", "training set")
                ),
                "validation": self._validation_loader(
                    stage.validation, ("batch", "validation", "validation set")
                ),
            }
        else:
            benchmark = None
        # training
        progress_epoch = Progress.new(schedule.epoch, f"epoch")
        logging.info(f"Start {stage.name}")
        start = datetime.now()
        for epoch in range(schedule.epoch):
            self.cleanup()
            Usage.checkpoint("classifier", stage.name, f"epoch{epoch}", "optimize")
            model.module.train()
            if len(bs.dataloader) > 0:
                progress_batch = Progress.new(len(bs.dataloader), "batch")
            for i, batch in enumerate(bs.dataloader):
                optimizer.zero_grad()
                loss = model.train(batch)
                loss.backward()
                optimizer.step()
                progress_batch.update(
                    i + 1, ("batch", "training", f"loss={loss.item():.4g}")
                )
            if benchmark is not None:
                Usage.checkpoint("classifier", stage.name, f"epoch{epoch}", "validate")
                model.module.eval()
                with torch.no_grad():
                    benchmark.append(
                        {"epoch": epoch}
                        | {k: model.validate(v) for k, v in validation_set.items()}
                    )
            lr.step()
            bs.step()
            if model.step is not NotImplemented:
                model.step()
            progress_epoch.update(epoch + 1)
            Usage.checkpoint("classifier", stage.name, f"epoch{epoch}", "finish")
        logging.info(
            f"{stage.name}: run {schedule.epoch} epochs in {datetime.now() - start}"
        )
        return benchmark

    @staticmethod
    def _validation_loader(dataset: Dataset, msg: MessageType = None):
        return simple_loader(
            dataset,
            report_progress=msg,
            batch_size=cfg.DataLoader.batch_eval,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    @torch.no_grad()
    def _evaluate(self, model: Model): ...  # TODO evaluation
