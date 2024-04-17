from __future__ import annotations

from abc import ABC, abstractmethod
from fractions import Fraction
from typing import TYPE_CHECKING

import torch
from base_class.math.random import SeedLike, Squares
from torch.utils.data import Dataset, Subset

from ..config.setting import torch as cfg
from ..utils import keep_fraction, noop
from . import Model

if TYPE_CHECKING:
    import numpy.typing as npt
    from torch import BoolTensor, Tensor


class Skimmer(Model):
    @property
    def n_parameters(self) -> int:
        return 0

    @property
    def module(self):
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
    def __init__(self, k: int, offset: int):
        self._k = k
        self._i = offset

    def split(self, batch: dict[str, Tensor]) -> tuple[BoolTensor, ...]:
        validation = torch.from_numpy((self._get_offset(batch) % self._k) == self._i)
        return ~validation, validation

    @classmethod
    def _get_offset(cls, batch: dict[str, Tensor]) -> npt.NDArray:
        return batch[cfg.KFold.offset].numpy().view(cfg.KFold.offset_dtype)


class RandomSubSample(KFold):
    def __init__(self, seed: SeedLike, fraction: str | Fraction):
        self._rng = Squares(seed)
        self._r = Fraction(fraction)

    def split(self, batch: dict[str, Tensor]) -> tuple[BoolTensor, ...]:
        training = torch.from_numpy(keep_fraction(self._r, self._random_offset(batch)))
        return training, ~training

    def _random_offset(self, batch: dict[str, Tensor]) -> npt.NDArray:
        offset = self._get_offset(batch)
        shape = offset.shape
        if len(shape) == 1:
            return self._rng.uint(offset)
        else:
            offset = offset.reshape(shape[0], -1)
            return self._rng.reduce(offset)


class RandomKFold(RandomSubSample):
    def __init__(self, seed: SeedLike, k: int, offset: int):
        self._rng = Squares(seed)
        self._k = k
        self._i = offset

    def split(self, batch: dict[str, Tensor]) -> tuple[BoolTensor, ...]:
        validation = torch.from_numpy((self._random_offset(batch) % self._k) == self._i)
        return ~validation, validation
