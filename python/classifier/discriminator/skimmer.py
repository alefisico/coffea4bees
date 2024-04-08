from __future__ import annotations

from abc import ABC, abstractmethod
from fractions import Fraction
from typing import TYPE_CHECKING

import torch
from base_class.math.random import SeedLike, Squares
from torch.utils.data import Dataset, Subset

from ..utils import keep_fraction, noop
from . import Model

if TYPE_CHECKING:
    from torch import BoolTensor, Tensor


class Skimmer(Model):
    @property
    def n_parameters(self) -> int:
        return 0

    @property
    def module(self):
        return noop

    def loss(self, _):
        return noop


class Splitter(ABC):
    @abstractmethod
    def split(self, batch: dict[str, Tensor]) -> tuple[BoolTensor, ...]: ...

    def setup(self, dataset: Dataset):
        self.reset()
        self._dataset = dataset

    def step(self, batch: dict[str, Tensor]) -> tuple[BoolTensor, ...]:
        selected = self.split(batch)
        size = len(selected[0])
        if self._selected is None:
            self._selected = torch.zeros(
                [len(selected), len(self._dataset)], dtype=torch.bool
            )
        for i, s in enumerate(selected):
            self._selected[i, self._start : self._start + size] = s
        self._start += size
        return selected

    def reset(self):
        self._dataset = None
        self._start = 0
        self._selected: BoolTensor = None

    def get(self):
        selected, dataset = self._selected, self._dataset
        self.reset()
        indices = torch.arange(len(dataset))
        return [Subset(dataset, indices[s]) for s in selected]


class KFold(Splitter):
    def __init__(self, k: int, offset: int, key: str):
        self._key = key

        self._k = k
        self._i = offset

    def split(self, batch: dict[str, Tensor]) -> tuple[BoolTensor, ...]:
        validation = (batch[self._key] % self._k) == self._i
        return ~validation, validation


class RandomSubSample(Splitter):
    def __init__(self, seed: SeedLike, fraction: str | Fraction, key: str):
        self._key = key

        self._rng = Squares(seed)
        self._r = Fraction(fraction)

    def split(self, batch: dict[str, Tensor]) -> tuple[BoolTensor, ...]:
        training = torch.from_numpy(
            keep_fraction(self._r, self._rng.uint(batch[self._key]))
        )
        return training, ~training


class RandomKFold(Splitter):
    def __init__(self, seed: SeedLike, k: int, offset: int, key: str):
        self._key = key

        self._rng = Squares(seed)
        self._k = k
        self._i = offset

    def split(self, batch: dict[str, Tensor]) -> tuple[BoolTensor, ...]:
        validation = torch.from_numpy(
            (self._rng.uint(batch[self._key]) % self._k) == self._i
        )
        return ~validation, validation
