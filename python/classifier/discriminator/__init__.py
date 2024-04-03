# TODO multiprocessing
from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from ..config.setting import default as Setting
from ..nn.dataset import mp_loader
from ..nn.schedule import Schedule
from ..process.device import Device
from ..typetools import WithUUID


@dataclass
class TrainingStage:
    name: str
    model: Model
    schedule: Schedule
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
    device: torch.device

    def __init__(self, **kwargs):
        super().__init__()
        self.metadata = kwargs
        self.name = "__".join(f"{k}_{v}" for k, v in kwargs.items())

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
        training: Dataset,
        validation: Dataset,
        device: Device,
    ):
        self.device = device.get(self.min_memory)
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
                training=training,
                validation=validation,
            )
        return result

    def eval(
        self,
    ): ...  # TODO evaluataion

    def _benchmark(self, epoch: int, pred: dict[str, Tensor], *group: str):
        return {}  # TODO monitor

    def _train(
        self,
        stage: TrainingStage,
        training: Dataset,
        validation: Dataset,
    ):
        model = stage.model
        schedule = stage.schedule
        # training preparation
        optimizer = schedule.optimizer(model.parameters())
        lr = schedule.lr_scheduler(optimizer)
        bs = schedule.bs_scheduler(
            training,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        # training
        benchmark = []
        datasets = {"training": training, "validation": validation}
        for epoch in range(schedule.epoch):
            self.cleanup()
            model.train()
            for batch in bs.dataloader:
                optimizer.zero_grad()
                pred = model.forward(batch)
                # TODO monitor pred
                loss = model.loss(pred)
                loss.backward()
                optimizer.step()
            if stage.do_benchmark:
                benchmark.append(
                    {
                        k: self._benchmark(
                            epoch,
                            self._evaluate(model, datasets[k]),
                            stage.name,
                            f"{k}-dataset",
                        )
                        for k in datasets
                    }
                )  # TODO monitor
            lr.step()
            bs.step()
            model.step()
        return benchmark

    @torch.no_grad()
    def _evaluate(
        self,
        model: Model,
        dataset: Dataset,
    ):
        loader = mp_loader(
            dataset,
            batch_size=Setting.DataLoader.batch_eval,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        model.eval()
        preds = [model.forward(batch) for batch in loader]
        return {k: torch.cat([p[k] for p in preds], dim=0) for k in preds[0]}
