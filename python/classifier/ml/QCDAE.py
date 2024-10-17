# QCDAE: setup training and validation steps
# see classifier.ml.HCR for more details
from __future__ import annotations

from typing import Iterable

import torch
import torch.types as tt
from torch import Tensor

from ..nn.blocks.QCDAE import PlaceholderBlock  # import the actual model here
from ..nn.schedule import Schedule
from . import BatchType
from .training import (
    BenchmarkStage,
    Model,
    MultiStageTraining,
    OutputStage,
    TrainingStage,
)


class QCDAEModel(Model):  # a wrapper for nn.Module
    def __init__(self, device: tt.Device, loss=...):
        self._nn = PlaceholderBlock(...)  # create an instance of QCDAE nn.Module
        self._loss = loss
        ...  # add other attributes e.g. loss

    @property
    def nn(self):  # an interface used by the training loop
        return self._nn

    def train(self, batch: BatchType) -> Tensor:
        ...  # the training loop, calculate the loss
        return self._loss(batch)  # return the loss

    def validate(
        self,
        batches: Iterable[
            BatchType
        ],  # a batch generator (actually a DataLoader) is passed instead of a single batch.
    ) -> dict[str]:
        # Unlike the training case, the loop is inside the function so that specific metrics can be accumulated.
        loss = 0
        for batch in batches:
            ...  # calculate some benchmark metrics e.g. loss, distribution of some variables
            loss += ...  # accumulate the loss
        return {"loss": loss}  # return a dictionary of metrics

    def step(self, epoch: int = None):
        # only necessary if some parameters are not taken care of by the scheduler.
        # something other than the batch size/learning rate e.g. ghost batch size
        ...


def _split_dataset(dataset):
    return ..., ...


class QCDAETraining(MultiStageTraining):
    def __init__(
        self,
        training_schedule: Schedule,
        arch_args=...,  # some arguments to specify the architecture/benchmarks/training parameters
        loss=...,  # some loss function
        **kwargs,
    ):
        # do not store large objects here, this object will be pickled and sent to other processes.
        super().__init__(
            **kwargs
        )  # pass some arguments to self.metadata, only used for naming/debugging
        self._arch = arch_args
        self._loss = loss
        self._training = training_schedule
        self._model: QCDAEModel = None  # do NOT initialize the model here

    def stages(self):
        # the following variables are set by the training loop before calling this function:
        self.dataset  # the whole dataset
        self.device  # the device to run the model
        # initialize the model here
        self._model = QCDAEModel(device=self.device, loss=self._loss)
        # self._model.to(self.device) # maybe needed if the code is not properly written
        # split the training and validation set
        training_set, validation_set = _split_dataset(self.dataset)
        yield BenchmarkStage(  # only do benchmark
            name="Baseline",  # do some baseline benchmarks before training
            model=self._model,
            validation=validation_set,
        )
        yield TrainingStage(  # do training and benchmark
            name="Train Dijet Layer",
            model=self._model,
            training=training_set,
            validation=validation_set,
            schedule=self._training,
        )
        # the model can be changed here e.g. freeze some layers/use a completely different model
        yield TrainingStage(
            name="Train Quadjet Layer",  # name is not required to be unique, but it is recommended
            model=self._model,
            training=training_set,
            validation=validation_set,
            schedule=self._training,
        )
        # save the model some where
        output = "path/to/output"
        with open(output, "wb") as f:
            torch.save(
                {
                    "model": self._model.nn.state_dict(),
                    "metadata": self.metadata,
                    "uuid": self.uuid,
                },
                f,
            )
        # save the path to output
        yield OutputStage(name="Final Model", path=output)
        # some clean up code can be added here if necessary
