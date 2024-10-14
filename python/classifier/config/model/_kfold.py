from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

from classifier.task import ArgParser, Model, converter, parse

if TYPE_CHECKING:
    from classifier.ml import MultiStageTraining
    from classifier.ml.skimmer import Splitter


class _KFold(Model):
    argparser = ArgParser()
    argparser.add_argument(
        "--kfolds",
        type=converter.bounded(int, lower=2),
        default=3,
        help="total number of folds",
    )
    argparser.add_argument(
        "--kfold-offsets",
        action="extend",
        nargs="+",
        default=[],
        help="selected offsets, e.g. [yellow]--kfold-offsets 0-3 5[/yellow]",
    )
    argparser.add_argument(
        "--kfold-seed",
        action="extend",
        nargs="+",
        default=["kfold"],
        help="the random seed to shuffle the dataset",
    )
    argparser.add_argument(
        "--kfold-seed-offsets",
        action="extend",
        nargs="+",
        default=[],
        help="the offsets to generate new seeds, e.g. [yellow]--kfold-seed-offsets 0-3 5[/yellow]. If not given, the dataset will not be shuffled.",
    )

    def _get_offsets(self, name: str, max: int = None) -> list[int]:
        offsets = getattr(self.opts, name)
        if not offsets:
            return [*range(max)] if max else []
        else:
            return parse.intervals(offsets, max)

    @cached_property
    def kfolds(self) -> int:
        return self.opts.kfolds

    @cached_property
    def offsets(self) -> list[int]:
        return self._get_offsets("kfold_offsets", self.kfolds)

    @cached_property
    def seeds(self) -> list[tuple[str]]:
        seed = self.opts.kfold_seed
        offsets = self._get_offsets("kfold_seed_offsets")
        if not offsets:
            return []
        return [(*seed, offset) for offset in offsets]


class KFoldClassifier(ABC, _KFold):
    @abstractmethod
    def initializer(self, splitter: Splitter, **kwargs) -> MultiStageTraining: ...

    def train(self):
        if not self.seeds:
            from classifier.ml.skimmer import KFold

            return [
                self.initializer(
                    KFold(self.kfolds, offset),
                    model=type(self).__name__,
                    kfolds=self.kfolds,
                    offset=offset,
                ).train
                for offset in self.offsets
            ]
        else:
            from classifier.ml.skimmer import RandomKFold

            return [
                self.initializer(
                    RandomKFold(seed, self.kfolds, offset),
                    model=type(self).__name__,
                    kfolds=self.kfolds,
                    offset=offset,
                    seed=seed,
                ).train
                for seed in self.seeds
                for offset in self.offsets
            ]
