import gc
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset

from ..dataset import NamedDataLoader
from ..static import Setting
from .schedule import Schedule


class Classifier(ABC):
    def __init__(self):
        self.model: nn.Module = None
        self.device = self._get_device()
        super().__init__()

    @staticmethod
    def _get_device():
        if Setting.use_cuda and torch.cuda.is_available():
            # TODO select device
            # cuda = torch.cuda.current_device()
            # device_name = torch.cuda.get_device_name(cuda)
            # device_count = torch.cuda.device_count()
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    @staticmethod
    def _cleanup():
        if Setting.use_cuda:
            torch.cuda.empty_cache()
        gc.collect()

    @property
    def n_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @abstractmethod
    def forward(self, batch: dict[str, Tensor], validation: bool = False):
        ...

    @abstractmethod
    def loss(self, batch: dict[str, Tensor], pred: Tensor) -> Tensor:
        ...

    def train(
            self,
            data: tuple[list[str], Dataset, Dataset],
            train: Schedule,
            finetune: Schedule = None):
        if train is not None:
            self._train_schedule(data, train)
        if finetune is not None:
            # TODO disable gradient for all but last layer
            self._train_schedule(data, finetune)

    def _train_schedule(
            self,
            dataset: tuple[list[str], Dataset, Dataset],
            schedule: Schedule):
        # training preparation
        optimizer = schedule.optimizer(self.model.parameters())
        lr_scheduler = schedule.lr_scheduler(optimizer)
        bs_scheduler = schedule.bs_scheduler(
            columns=dataset[0],
            dataset=dataset[1],
            num_workers=schedule.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        # validation preparation
        validation = NamedDataLoader(
            columns=dataset[0],
            dataset=dataset[2],
            batch_size=schedule.bs_eval,
            num_workers=schedule.num_workers,
            shuffle=False,
            pin_memory=True)

        # training
        for epoch in range(schedule.epoch):  # TODO start from intermediate epoch
            batch, pred, loss = None, None, None
            self._cleanup()
            for batch in bs_scheduler.dataloader:
                optimizer.zero_grad()
                pred = self.forward(batch)
                loss = self.loss(batch, pred)
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            bs_scheduler.step()

    def eval(
            self,
            data):
        ...
