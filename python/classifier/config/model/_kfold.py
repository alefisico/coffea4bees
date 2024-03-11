from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

from classifier.nn.dataset import io_loader
from classifier.task import ArgParser, Model, converter

from ..setting.df import Columns

if TYPE_CHECKING:
    from classifier.discriminator import Classifier
    from classifier.process.device import Device
    from torch.utils.data import Dataset, StackDataset


class KFoldClassifier(ABC, Model):
    argparser = ArgParser()
    argparser.add_argument(
        "--kfolds",
        type=converter.int_pos,
        default=3,
        help="total number of folds",
    )
    argparser.add_argument(
        "--kfold-max-folds",
        type=converter.int_pos,
        default=argparse.SUPPRESS,
        help="the maximum number of folds to use",
    )
    argparser.add_argument(
        "--kfold-split-key",
        default=argparse.SUPPRESS,
        help="the key used to split the dataset (default: [green]Columns[/green].event_offset)",
    )

    @abstractmethod
    def initializer(self, kfolds: int, offset: int) -> Classifier: ...

    @cached_property
    def kfolds(self) -> int:
        return self.opts.kfolds

    @cached_property
    def kfold_key(self) -> str:
        if not hasattr(self.opts, "kfold_split_key"):
            return Columns.event_offset
        return self.opts.kfold_split_key

    def train(self, dataset: StackDataset):
        if self.kfolds == 1:
            return [
                _train_classifier(
                    self.initializer(kfolds=self.kfolds, offset=0), dataset, dataset
                )
            ]
        else:
            import numpy as np
            from torch.utils.data import ConcatDataset, Subset

            key = dataset.datasets[self.kfold_key]
            offset = []
            for i in io_loader(key):
                offset.append(i.numpy() % self.kfolds)
            offset = np.concatenate(offset)
            indices = np.arange(len(offset))
            folds = [Subset(dataset, indices[offset == i]) for i in range(self.kfolds)]
            max_folds = min(
                self.kfolds, getattr(self.opts, "kfold_max_folds", self.kfolds)
            )
            return [
                _train_classifier(
                    self.initializer(kfolds=self.kfolds, offset=i),
                    ConcatDataset(folds[:i] + folds[i + 1 :]),
                    folds[i],
                )
                for i in range(max_folds)
            ]


class _train_classifier:
    def __init__(
        self,
        classifier: Classifier,
        training: Dataset,
        validation: Dataset,
    ):
        self._classifier = classifier
        self._training = training
        self._validation = validation

    def __call__(self, device: Device):
        return self._classifier.train(
            training=self._training, validation=self._validation, device=device
        )
