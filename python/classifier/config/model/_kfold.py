from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

from classifier.nn.dataset import io_loader
from classifier.task import ArgParser, Model, converter, parse

if TYPE_CHECKING:
    from classifier.discriminator import Classifier
    from classifier.process.device import Device
    from torch.utils.data import Dataset, StackDataset


class _KFold(Model):
    argparser = ArgParser()
    argparser.add_argument(
        "--kfolds",
        type=converter.int_pos,
        default=3,
        help="total number of folds",
    )
    argparser.add_argument(
        "--kfold-offsets",
        action="extend",
        nargs="+",
        default=[],
        help="selected offsets e.g. [yellow]--chunks 0-3 5[/yellow]",
    )
    argparser.add_argument(
        "--kfold-split-key",
        default="offset",
        help="the key used to split the dataset",
    )

    @cached_property
    def kfolds(self) -> int:
        return self.opts.kfolds

    @cached_property
    def offsets(self) -> list[int]:
        offsets = getattr(self.opts, "kfold_offsets", None)
        if offsets is None:
            offsets = [*range(self.kfolds)]
        else:
            offsets = parse.intervals(offsets, self.kfolds)
        return offsets


class KFoldClassifier(ABC, _KFold):
    @abstractmethod
    def initializer(self, kfolds: int, offset: int) -> Classifier: ...

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

            key = dataset.datasets[self.opts.kfold_split_key]
            offset = []
            for i in io_loader(key):
                offset.append(i.numpy() % self.kfolds)
            offset = np.concatenate(offset)
            indices = np.arange(len(offset))
            folds = [Subset(dataset, indices[offset == i]) for i in range(self.kfolds)]
            return [
                _train_classifier(
                    self.initializer(kfolds=self.kfolds, offset=i),
                    ConcatDataset(folds[:i] + folds[i + 1 :]),
                    folds[i],
                )
                for i in self.offsets
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
