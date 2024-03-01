# TODO testing with new data
# TODO multiprocessing
from __future__ import annotations

import gc
import uuid
from abc import ABC, abstractmethod
from functools import cached_property, reduce
from typing import Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ..config.setting.default import DataLoader as DLSetting
from ..config.state.monitor import Classifier as Monitor
from ..nn.schedule import Schedule
from ..process.device import Device


class Classifier(ABC):
    device: torch.device

    def __init__(self, **kwargs):
        self.metadata = kwargs
        self.name = '__'.join(f'{k}_{v}' for k, v in kwargs.items())
        self.uuid = uuid.uuid1()

    def cleanup(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    @cached_property
    def n_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @cached_property
    def min_memory(self):
        return 0

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        ...

    @property
    @abstractmethod
    def train_schedule(self) -> Schedule:
        ...

    @property
    def finetune_schedule(self) -> Optional[Schedule]:
        ...

    def setup_finetune(self):  # TODO modify
        ...

    @abstractmethod
    def forward(self, batch: dict[str, Tensor], validation: bool = False) -> dict[str, Tensor]:
        ...

    @abstractmethod
    def loss(self, pred: dict[str, Tensor]) -> Tensor:
        ...

    def train(
        self,
        training: Dataset,
        validation: Dataset,
        device: Device,
    ):
        self.device = device.get(self.min_memory)
        result = {
            'uuid': str(self.uuid),
            'name': self.name,
            'metadata': self.metadata,
            'parameters': self.n_trainable_parameters,
            'benchmark': {},
        }
        result['benchmark']['train'] = self._train(
            training, validation, self.train_schedule, 'training steps')
        if self.finetune_schedule is not None:
            self.setup_finetune()
            result['benchmark']['finetune'] = self._train(
                training, validation, self.finetune_schedule, 'finetuning steps')
        return result

    def eval(
        self,
    ):
        ...  # TODO evaluataion

    def _benchmark(self, epoch: int, pred: dict[str, Tensor], *group: str):
        return reduce(dict.__or__, (f(
            pred=pred,
            uuid=self.uuid,
            name=self.name,
            epoch=epoch,
            group=group,
        ) for f in Monitor.benchmark), {})

    def _train(
        self,
        training: Dataset,
        validation: Dataset,
        schedule: Schedule,
        name: str,
    ):
        # training preparation
        optimizer = schedule.optimizer(self.model.parameters())
        lr = schedule.lr_scheduler(optimizer)
        bs = schedule.bs_scheduler(
            dataset=training,
            num_workers=DLSetting.num_workers,
            shuffle=True,
            drop_last=True)

        # training
        benchmark = []
        datasets = {'training': training, 'validation': validation}
        for epoch in range(schedule.epoch):
            self.cleanup()
            self.model.train()
            for batch in bs.dataloader:
                optimizer.zero_grad()
                pred = self.forward(batch)
                loss = self.loss(pred)
                loss.backward()
                optimizer.step()
            benchmark.append({
                k: self._benchmark(
                    epoch, self._evaluate(datasets[k]), name, f'{k} set'),
            } for k in datasets)
            lr.step()
            bs.step()
        return benchmark

    @torch.no_grad()
    def _evaluate(
        self,
        validation: Dataset,
    ):
        loader = DataLoader(
            validation,
            batch_size=DLSetting.batch_eval,
            num_workers=DLSetting.num_workers,
            shuffle=False,
            drop_last=False)
        self.model.eval()
        preds = [self.forward(batch, validation=True) for batch in loader]
        return {
            k: torch.cat([p[k] for p in preds], dim=0)
            for k in preds[0]}
