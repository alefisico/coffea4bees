from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from ..config.setting import torch as cfg
from ..monitor.progress import Progress
from ..nn.dataset import mp_loader
from ..nn.schedule import Schedule
from ..process.device import Device
from ..task.special import interface
from ..typetools import WithUUID


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

    def to(self, device: torch.device):
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

    @interface
    def evaluate(self, batch: dict[str, Tensor]) -> dict[str, Tensor]: ...

    @interface(optional=True)
    def step(self, epoch: int = None): ...


class Classifier(WithUUID, ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.metadata = kwargs
        self.name = "__".join(f"{k}_{v}" for k, v in kwargs.items())
        self.device: torch.device = None
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
            "parameters": {},
            "benchmark": {},
        }
        for stage in self.training_stages():
            result["parameters"][stage.name] = stage.model.n_parameters
            result["benchmark"][stage.name] = self._train(
                stage=stage,
            )
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

        # training
        benchmark = []
        epoch_p = Progress.new(schedule.epoch, f"epoch")
        logging.info(f"Start {stage.name}")
        start = datetime.now()
        for epoch in range(schedule.epoch):
            self.cleanup()
            model.module.train()
            batch_p = Progress.new(len(bs.dataloader), "batch")
            for i, batch in enumerate(bs.dataloader):
                optimizer.zero_grad()
                loss = model.train(batch)
                loss.backward()
                optimizer.step()
                batch_p.update(i + 1, f"batch|loss={loss.item():.4g}")
            if (
                (not cfg.Training.disable_benchmark)
                and stage.do_benchmark
                and (model.validate is not NotImplemented)
            ):
                benchmark.append(
                    {
                        "training": self._validate(model, stage.training),
                        "validation": self._validate(model, stage.validation),
                    }
                )
            lr.step(), bs.step()
            if model.step is not NotImplemented:
                model.step()
            epoch_p.update(epoch + 1)
        logging.info(
            f"{stage.name}: run {schedule.epoch} epochs in {datetime.now() - start}"
        )
        return benchmark

    @torch.no_grad()
    def _validate(
        self,
        model: Model,
        dataset: Dataset,
    ):
        model.module.eval()
        return model.validate(
            mp_loader(
                dataset,
                batch_size=cfg.DataLoader.batch_eval,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )
        )

    @torch.no_grad()
    def _evaluate(self, model: Model): ...  # TODO evaluation
