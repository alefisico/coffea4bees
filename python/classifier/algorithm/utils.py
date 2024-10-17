from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from numbers import Number

    import numpy.typing as npt

    from ..ml import BatchType


def to_num(tensor: torch.Tensor):
    return tensor.item()


def to_arr(tensor: torch.Tensor) -> npt.NDArray:
    return tensor.numpy(force=True)


class Selector:
    def __init__(self, selection: torch.BoolTensor):
        self._selection = selection

    def select(self, batch: BatchType) -> BatchType:
        return {k: v[self._selection] for k, v in batch.items()}

    def pad(
        self,
        batch: BatchType,
        values: Number | dict[torch.dtype | None, Number] = torch.nan,
    ):
        length = self._selection.shape[0]
        padded = {}
        for k, v in batch.items():
            if not isinstance(values, dict):
                value = values
            else:
                value = values.get(v.dtype, values.get(None, torch.nan))
            new = torch.full(
                (length, *v.shape[1:]), value, dtype=v.dtype, device=v.device
            )
            new[self._selection] = v
            padded[k] = new
        return padded
