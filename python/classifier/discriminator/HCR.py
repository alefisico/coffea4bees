from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable

import fsspec
import torch
import torch.nn.functional as F
import torch.types as tt
from torch import Tensor

from ..algorithm.utils import to_num
from ..config import setting as cfg
from ..config.scheduler import SkimStep
from ..config.setting.HCR import Input, InputBranch, Output
from ..config.state.label import MultiClass
from ..nn.HCR_blocks import HCR
from ..nn.schedule import MilestoneStep, Schedule
from . import BatchType, BenchmarkStage, Classifier, Model, OutputStage, TrainingStage
from .roc import MulticlassROC
from .skimmer import Skimmer, Splitter


@dataclass
class HCRArch:
    loss: Callable[[BatchType], Tensor]
    n_features: int = 8
    use_attention_block: bool = True


@dataclass
class GBNSchedule(MilestoneStep):
    n_batches: int = 64
    milestones: list[int] = (1, 3, 6, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
    gamma: float = 0.25

    def __post_init__(self):
        self.milestones = sorted(self.milestones)
        self._last_bs = self.n_batches
        self.reset()

    def get_bs(self):
        self._last_bs = int(self.n_batches * (self.gamma**self.milestone))
        return self._last_bs

    def get_last_bs(self):
        return self._last_bs


@dataclass
class HCRBenchmarks:
    rocs: Iterable[MulticlassROC]


def _HCRInput(batch: BatchType, device: tt.Device, selection: Tensor = None):
    for k, v in batch.items():
        batch[k] = v.to(device, non_blocking=True)
    inputs = [batch.pop(k) for k in (Input.CanJet, Input.NotCanJet, Input.ancillary)]
    if selection is not None:
        selection = selection.to(device, non_blocking=True)
        inputs = [i[selection] for i in inputs]
    return inputs


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
    def train(self, batch: BatchType):
        training, _ = self._splitter.step(batch)
        self._nn.updateMeanStd(*_HCRInput(batch, self._device, training))
        # TODO compute die loss
        return super().train(batch)


class HCRModel(Model):
    def __init__(
        self,
        device: tt.Device,
        arch: HCRArch,
        benchmarks: HCRBenchmarks,
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
        self._benchmarks = benchmarks

    @property
    def ghost_batch(self):
        return self._gbn

    @ghost_batch.setter
    def ghost_batch(self, gbn: GBNSchedule):
        self._gbn = gbn
        if gbn is None:
            self._nn.setGhostBatches(0, False)
        else:
            self._gbn.reset()
            self._nn.setGhostBatches(self._gbn.n_batches, True)  # TODO check subset

    @property
    def hyperparameters(self) -> dict[str]:
        return {"n_ghost_batch": self.ghost_batch.get_last_bs()}

    @property
    def nn(self):
        return self._nn

    def train(self, batch: BatchType) -> Tensor:
        c, p = self._nn(*_HCRInput(batch, self._device))
        batch[Output.quadjet_raw] = p
        batch[Output.class_raw] = c
        return self._loss(batch)

    def validate(self, batches: Iterable[BatchType]) -> dict[str]:
        loss, weight = 0.0, 0.0
        rocs = [r.copy() for r in self._benchmarks.rocs]
        for batch in batches:
            c, p = self._nn(*_HCRInput(batch, self._device))
            batch |= {
                Output.quadjet_raw: p,
                Output.class_raw: c,
                Output.class_prob: F.softmax(c, dim=1),
            }
            sumw = to_num(batch[Input.weight].sum())
            loss += to_num(self._loss(batch)) * sumw
            weight += sumw
            for roc in rocs:
                roc.update(batch)
        return {"loss": loss / weight, "roc": [r.roc() for r in rocs]}

    def step(self, epoch: int = None):
        if self.ghost_batch is not None and self.ghost_batch.step(epoch):
            self._nn.setGhostBatches(self.ghost_batch.get_bs(), True)


class HCRTraining(Classifier):
    def __init__(
        self,
        arch: HCRArch,
        ghost_batch: GBNSchedule,
        cross_validation: Splitter,
        training_schedule: Schedule,
        finetuning_schedule: Schedule = None,
        benchmarks: HCRBenchmarks = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._arch = arch
        self._ghost_batch = ghost_batch
        self._splitter = cross_validation
        self._training = training_schedule
        self._finetuning = finetuning_schedule
        self._benchmarks = benchmarks or HCRBenchmarks()
        self._HCR: HCRModel = None

    def stages(self):
        self._HCR = HCRModel(
            device=self.device,
            arch=self._arch,
            benchmarks=self._benchmarks,
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
        )
        self._HCR.nn.initMeanStd()
        training, validation = self._splitter.get()
        validation = {
            "training": training,
            "validation": validation,
        }
        yield BenchmarkStage(
            name="Baseline",
            model=self._HCR,
            validation=validation,
        )
        yield TrainingStage(
            name="Training",
            model=self._HCR,
            schedule=self._training,
            training=training,
            validation=validation,
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
            )
            self._HCR.ghost_batch = self._ghost_batch
            layers.setLayerRequiresGrad(requires_grad=True)
        output = cfg.IO.output / f"{self.name}__{self.uuid}.pkl"
        if not output.is_null:
            logging.info(f"Saving model to {output}")
            with fsspec.open(output, "wb") as f:
                torch.save(
                    {
                        "model": self._HCR.nn.state_dict(),
                        "metadata": self.metadata,
                        "uuid": self.uuid,
                        "label": MultiClass.labels,
                    },
                    f,
                )
            yield OutputStage(name="Final", path=output)
