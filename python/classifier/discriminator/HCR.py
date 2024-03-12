from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Callable

import torch

from ..config.scheduler import SkimStep
from ..config.setting.HCR import Architecture, Input, InputBranch, Output
from ..config.state.label import MultiClass
from ..nn.blocks import HCR
from ..utils import noop
from . import Classifier, Module, TrainingStage

if TYPE_CHECKING:
    from torch import Tensor

    from ..nn.schedule import Schedule


class _SetupGBN(Module):
    def __init__(self, device: torch.device):
        self.tensors = defaultdict(list)
        self.device = device

    @property
    def n_parameters(self) -> int:
        return 0

    @property
    def module(self):
        return noop()

    def forward(self, batch: dict[str, Tensor]):
        for k in [Input.CanJet, Input.NotCanJet, Input.ancillary]:
            self.tensors[k].append(batch[k].to(self.device))
        return {}

    def loss(self, _):
        return noop()


class HCRModule(Module):

    def __init__(
        self,
        loss: Callable[[HCRModule, dict[str, Tensor]], Tensor],
        device: torch.device,
        gbn_size: int = 0,
    ):
        self._nn = HCR(
            dijetFeatures=Architecture.n_features,
            quadjetFeatures=Architecture.n_features,
            ancillaryFeatures=InputBranch.feature_ancillary,
            useOthJets=(
                "attention" if Architecture.use_attention_block else ""
            ),  # TODO take from init arg
            device=device,
            nClasses=len(MultiClass.labels),
        )
        self._nn.setGhostBatches(gbn_size, gbn_size > 0)
        self._loss = loss
        self._device = device

    @property
    def module(self):
        return self._nn

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        CanJet = batch.pop(Input.CanJet)
        NotCanJet = batch.pop(Input.NotCanJet)
        ancillary = batch.pop(Input.ancillary)
        c, p = self._nn(
            CanJet.to(self._device),
            NotCanJet.to(self._device),
            ancillary.to(self._device),
        )
        batch[Output.quadjet_score] = p
        batch[Output.class_score] = c
        for k, v in batch.items():
            batch[k] = v.to(self._device, non_blocking=True)
        return batch

    def loss(self, pred: dict[str, Tensor]) -> Tensor:
        return self._loss(self, pred)


class HCRClassifier(Classifier):

    def __init__(
        self,
        loss: Callable[[dict[str, Tensor]], Tensor],
        training_schedule: Schedule,
        finetuning_schedule: Schedule = None,
        ghost_batch_size: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._loss = loss
        self._training = training_schedule
        self._finetuning = finetuning_schedule
        self._gbn_size = ghost_batch_size
        self._module: HCRModule = None

    def training_stages(self):
        gbn = _SetupGBN(self.device)
        yield TrainingStage(
            name="Setup GBN",
            module=gbn,
            schedule=SkimStep(),
            do_benchmark=False,
        )
        if self._module is None:
            self._module = HCRModule(
                loss=self._loss,
                device=self.device,
                gbn_size=self._gbn_size,
            )  # TODO config ghost batch size
            self._module.to(self.device)
            self._module.module.setMeanStd(
                torch.cat(gbn.tensors[Input.CanJet]),
                torch.cat(gbn.tensors[Input.NotCanJet]),
                torch.cat(gbn.tensors[Input.ancillary]),
            )
            gbn = None
        yield TrainingStage(
            name="Training",
            module=self._module,
            schedule=self._training,
            do_benchmark=True,
        )
        # TODO finetuning
