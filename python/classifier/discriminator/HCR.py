from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable

import fsspec
import numpy as np
import torch
import torch.nn.functional as F
import torch.types as tt
from torch import Tensor

from ..algorithm.utils import to_arr, to_num
from ..config import setting as cfg
from ..config.scheduler import SkimStep
from ..config.setting.HCR import Input, InputBranch, Output
from ..config.state.label import MultiClass
from ..nn.HCR_blocks import HCR
from ..nn.schedule import MilestoneStep, Schedule
from . import Classifier, Model, TrainingStage
from .skimmer import Skimmer, Splitter


@dataclass
class HCRArch:
    loss: Callable[[dict[str, Tensor]], Tensor]
    n_features: int = 8
    use_attention_block: bool = True


@dataclass
class GBN(MilestoneStep):
    n_batches: int = 64
    milestones: list[int] = (1, 3, 6, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
    gamma: float = 0.25

    def __post_init__(self):
        self.milestones = sorted(self.milestones)
        self.reset()


def _HCRInput(batch: dict[str, Tensor], device: tt.Device, selection: Tensor = None):
    CanJet = batch.pop(Input.CanJet)
    NotCanJet = batch.pop(Input.NotCanJet)
    ancillary = batch.pop(Input.ancillary)
    if selection is not None:
        CanJet = CanJet[selection]
        NotCanJet = NotCanJet[selection]
        ancillary = ancillary[selection]
    return CanJet.to(device), NotCanJet.to(device), ancillary.to(device)


class _HCRSkim(Skimmer):
    def __init__(
        self,
        nn: HCR,
        device: tt.Device,
        splitter: Splitter,
    ):
        self._nn = nn
        self._device = device
        self._splitter = splitter

    @torch.no_grad()
    def train(self, batch: dict[str, Tensor]):
        training, _ = self._splitter.step(batch)
        self._nn.updateMeanStd(*_HCRInput(batch, self._device, training))
        # TODO compute die loss
        return super().train(batch)


class HCRModel(Model):
    def __init__(
        self,
        arch: HCRArch,
        device: tt.Device,
    ):
        self._loss = arch.loss
        self._device = device
        self._gbn = None
        self._nn = HCR(
            dijetFeatures=arch.n_features,
            quadjetFeatures=arch.n_features,
            ancillaryFeatures=InputBranch.feature_ancillary,
            useOthJets=("attention" if arch.use_attention_block else ""),
            device=device,
            nClasses=len(MultiClass.labels),
        )

    @property
    def ghost_batch(self):
        return self._gbn

    @ghost_batch.setter
    def ghost_batch(self, gbn: GBN):
        self._gbn = gbn
        if gbn is None:
            self._nn.setGhostBatches(0, False)
        else:
            self._gbn.reset()
            self._nn.setGhostBatches(self._gbn.n_batches, True)  # TODO check subset

    @property
    def module(self):
        return self._nn

    def train(self, batch: dict[str, Tensor]) -> Tensor:
        c, p = self._nn(*_HCRInput(batch, self._device))
        batch[Output.quadjet_score] = p
        batch[Output.class_score] = c
        for k, v in batch.items():
            batch[k] = v.to(self._device, non_blocking=True)
        return self._loss(batch)

    def step(self, epoch: int = None):
        if self.ghost_batch is not None and self.ghost_batch.step(epoch):
            self._nn.setGhostBatches(
                int(
                    self.ghost_batch.n_batches
                    * (self.ghost_batch.gamma**self.ghost_batch.milestone)
                ),
                True,
            )


class HCRClassifier(Classifier):
    def __init__(
        self,
        arch: HCRArch,
        ghost_batch: GBN,
        cross_validation: Splitter,
        training_schedule: Schedule,
        finetuning_schedule: Schedule = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._arch = arch
        self._ghost_batch = ghost_batch
        self._splitter = cross_validation
        self._training = training_schedule
        self._finetuning = finetuning_schedule
        self._HCR: HCRModel = None

    def training_stages(self):
        self._HCR = HCRModel(
            arch=self._arch,
            device=self.device,
        )
        self._HCR.ghost_batch = self._ghost_batch
        self._HCR.to(self.device)
        self._splitter.setup(self.dataset)
        skim = _HCRSkim(self._HCR._nn, self.device, self._splitter)
        yield TrainingStage(
            name="Initialization",
            model=skim,
            schedule=SkimStep(),
            training=self.dataset,
            do_benchmark=False,
        )
        self._HCR.module.initMeanStd()
        training, validation = self._splitter.get()
        yield TrainingStage(
            name="Training",
            model=self._HCR,
            schedule=self._training,
            training=training,
            validation=validation,
            do_benchmark=True,
        )
        self._HCR.ghost_batch = None
        if self._finetuning is not None:
            layers = self._HCR._nn.layers
            layers.setLayerRequiresGrad(
                requires_grad=False, index=sorted(layers.layers.keys())[:-1]
            )
            yield TrainingStage(
                name="Finetuning",
                model=self._HCR,
                schedule=self._finetuning,
                training=training,
                validation=validation,
                do_benchmark=True,
            )
            self._HCR.ghost_batch = self._ghost_batch
            layers.setLayerRequiresGrad(requires_grad=True)
        output = cfg.IO.output / f"{self.name}__{self.uuid}.pkl"
        if not output.is_null:
            logging.info(f"Saving model to {output}")
            with fsspec.open(output, "wb") as f:
                torch.save(
                    {
                        "model": self._HCR.module.state_dict(),
                        "metadata": self.metadata,
                        "uuid": self.uuid,
                        "label": MultiClass.labels,
                    },
                    f,
                )
