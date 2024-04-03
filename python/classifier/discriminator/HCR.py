from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import fsspec
import torch

from ..config.scheduler import SkimStep
from ..config.setting.default import IO as IOSetting
from ..config.setting.HCR import Input, InputBranch, Output
from ..config.state.label import MultiClass
from ..nn.blocks import HCR
from ..nn.schedule import MilestoneStep
from ..utils import noop
from . import Classifier, Model, TrainingStage

if TYPE_CHECKING:
    from torch import Tensor

    from ..nn.schedule import Schedule


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


def _HCRInput(batch: dict[str, Tensor], device: torch.device):
    CanJet = batch.pop(Input.CanJet)
    NotCanJet = batch.pop(Input.NotCanJet)
    ancillary = batch.pop(Input.ancillary)
    return CanJet.to(device), NotCanJet.to(device), ancillary.to(device)


class _HCRSkim(Model):
    def __init__(self, nn: HCR, device: torch.device):
        self._nn = nn
        self._device = device

    @property
    def n_parameters(self) -> int:
        return 0

    @property
    def module(self):
        return noop

    def forward(self, batch: dict[str, Tensor]):
        self._nn.updateMeanStd(*_HCRInput(batch, self._device))

    def loss(self, _):
        return noop


class HCRModel(Model):
    def __init__(
        self,
        arch: HCRArch,
        device: torch.device,
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
            self._nn.setGhostBatches(
                self._gbn.n_batches, True
            )  # TODO check what the subset do

    @property
    def module(self):
        return self._nn

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        c, p = self._nn(*_HCRInput(batch, self._device))
        batch[Output.quadjet_score] = p
        batch[Output.class_score] = c
        for k, v in batch.items():
            batch[k] = v.to(self._device, non_blocking=True)
        return batch

    def loss(self, pred: dict[str, Tensor]) -> Tensor:
        return self._loss(self, pred)

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
        training_schedule: Schedule,
        finetuning_schedule: Schedule = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._arch = arch
        self._ghost_batch = ghost_batch
        self._training = training_schedule
        self._finetuning = finetuning_schedule
        self._HCR: HCRModel = None

    def training_stages(self):
        if self._HCR is None:
            self._HCR = HCRModel(
                arch=self._arch,
                device=self.device,
            )
            self._HCR.ghost_batch = self._ghost_batch
            self._HCR.to(self.device)
            skim = _HCRSkim(self._HCR._nn, self.device)
            yield TrainingStage(
                name="Initialization",
                model=skim,
                schedule=SkimStep(),
                do_benchmark=False,
            )
            self._HCR.module.initMeanStd()
        yield TrainingStage(
            name="Training",
            model=self._HCR,
            schedule=self._training,
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
                do_benchmark=True,
            )
            self._HCR.ghost_batch = self._ghost_batch
            layers.setLayerRequiresGrad(requires_grad=True)
        with fsspec.open(IOSetting.output / f"{self.name}_{self.uuid}.pkl", "wb") as f:
            torch.save(
                {
                    "model": self._HCR.module.state_dict(),
                    "metadata": self.metadata,
                    "uuid": self.uuid,
                },
                f,
            )
