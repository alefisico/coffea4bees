from numbers import Real
from typing import Iterable

import numpy as np
import torch
import torch.types as tt
from base_class.typetools import check_type
from torch import Tensor

from ...utils import not_none

RegularAxis = tuple[int, Real, Real]


class FloatWeighted:
    def __init__(
        self,
        bins: RegularAxis | Iterable[float],
        err: bool = True,
        dtype: tt._dtype = None,
        device: tt.Device = None,
    ):
        self._err = err
        self._dtype = dtype
        self._device = device

        self._regular = check_type(bins, RegularAxis)
        if self._regular:
            step = (bins[2] - bins[1]) / bins[0]
            self._edge = (bins[0], step, bins[1] / step - 1)
            self._nbin = bins[0] + 2
        else:
            self._edge = np.sort(bins)
            self._nbin = len(self._edge) + 1

        self.reset()

    def copy(self, dtype: tt._dtype = None, device: tt.Device = None):
        new = super().__new__(self.__class__)
        new.__dict__ = self.__dict__.copy()
        new.reset()
        if dtype is not None:
            new._dtype = dtype
        if device is not None:
            new._device = device
        return new

    def reset(self):
        self.__bins: Tensor = None
        self.__errs: Tensor = None
        self.__edge: Tensor = None

    def _init(self, x: Tensor, weight: Tensor):
        _b = dict(
            dtype=not_none(self._dtype, weight.dtype),
            device=not_none(self._device, weight.device),
        )
        self.__bins = torch.zeros(self._nbin, **_b)
        if self._err:
            self.__errs = torch.zeros(self._nbin, **_b)
        _e = dict(
            dtype=not_none(self._dtype, x.dtype),
            device=not_none(self._device, x.device),
        )
        if self._regular:
            self.__edge = torch.tensor(self._edge[1:], **_e)
        else:
            self.__edge = torch.as_tensor(self._edge, **_e)

    def fill(self, x: Tensor, weight: Tensor):
        # -inf < b[0] < e[0] <= b[1] < e[1] <= ... < e[-1] <= b[-1] < inf
        if self.__bins is None:
            self._init(x=x, weight=weight)
        e = self.__edge
        if self._regular:
            indices = torch.clip(x / e[0] - e[1], 0, self._edge[0] + 1).to(torch.int32)
        else:
            indices = torch.bucketize(x, e, right=True, out_int32=True)
        self.__bins.index_add_(0, indices, weight)
        if self._err:
            self.__errs.index_add_(0, indices, weight**2)

    def values(self):
        return self.__bins

    def errors(self):
        return torch.sqrt(self.__errs)

    def __repr__(self):
        if self._regular:
            start = (self._edge[2] + 1) * self._edge[1]
            end = start + self._edge[0] * self._edge[1]
            edges = np.linspace(start, end, self._edge[0] + 1)
        else:
            edges = self._edge
        prev = "-\u221E"
        lines = []
        errs = self.errors() if self._err else None
        for i, edge in enumerate([*map("{:.6g}".format, edges), "\u221E"]):
            line = "(" if i == 0 else "["
            line += f"{prev}, {edge}) {self.__bins[i]:.6g}"
            if self._err:
                line += f" \u00B1 {errs[i]:.6g}"
            prev = edge
            lines.append(line)
        return "\n".join(lines)
