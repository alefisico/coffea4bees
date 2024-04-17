from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from ..config.setting.torch import DataLoader as cfg
from ..monitor.progress import Progress
from ..nn.dataset import mp_loader
from ..nn.schedule import Schedule
from ..process.device import Device
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
    def eval(self):
        self.module.eval()

    def train(self):
        self.module.train()

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

    @abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]: ...

    @abstractmethod
    def loss(self, pred: dict[str, Tensor]) -> Tensor: ...

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

    def eval(
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
        for epoch in range(schedule.epoch):
            self.cleanup()
            model.train()
            batch_p = Progress.new(len(bs.dataloader), "batch")
            for i, batch in enumerate(bs.dataloader):
                optimizer.zero_grad()
                pred = model.forward(batch)
                loss = model.loss(pred)
                loss.backward()
                optimizer.step()
                batch_p.update(i + 1, f"batch|loss={loss.item():.4g}")
            if stage.do_benchmark:
                ...  # TODO benchmark
            lr.step()
            bs.step()
            model.step()
            epoch_p.update(epoch + 1)
        return benchmark

    @torch.no_grad()
    def _evaluate(
        self,
        model: Model,
        dataset: Dataset,
    ):
        loader = mp_loader(
            dataset,
            batch_size=cfg.batch_eval,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        model.eval()
        preds = [model.forward(batch) for batch in loader]
        ...  # TODO evaluation
