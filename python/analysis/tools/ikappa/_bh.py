from __future__ import annotations

from itertools import repeat
from typing import Iterable, overload

import numpy as np
import numpy.typing as npt
from hist import Hist
from hist.axis import (
    Boolean,
    IntCategory,
    Integer,
    Regular,
    StrCategory,
    Variable,
)

INTERNAL = '<span style="font-style: italic; font-weight: bold;">{text}</span>'.format
HistAxis = Boolean | IntCategory | Integer | Regular | StrCategory | Variable


class BHAxis:
    _TRUE = "True"
    _FALSE = "False"
    _OTHERS = INTERNAL(text="Others")

    def __init__(self, flow: bool, floating_format="{:.6g}".format):
        self._flow = flow
        self._ff = floating_format

    def flow(self, axis: HistAxis) -> tuple[bool, bool]:
        if not self._flow:
            return False, False
        ax = axis.traits
        return ax.underflow, ax.overflow

    def widths(self, axis: HistAxis) -> list[float] | None:
        if not isinstance(axis, (Regular, Variable)):
            return None
        under, over = self.flow(axis)
        width = [*axis.widths]
        if under:
            width.insert(0, np.inf)
        if over:
            width.append(np.inf)
        return width

    def edges(self, axis: Regular | Variable, finite: bool = False) -> list[float]:
        under, over = self.flow(axis)
        edges = [*axis.edges]
        if under or over:
            flow = np.min(axis.widths) if finite else np.inf
            if under:
                edges.insert(0, edges[0] - flow)
            if over:
                edges.append(edges[-1] + flow)
        return edges

    def labels(self, axis: HistAxis) -> list[str]:
        under, over = self.flow(axis)
        cats = [*axis]
        match axis:
            case Boolean():
                return [self._FALSE, self._TRUE]
            case IntCategory():
                return [*map(str, cats)] + ([self._OTHERS] if over else [])
            case StrCategory():
                return cats + ([self._OTHERS] if over else [])
            case Integer():
                return (
                    ([f"<{cats[0]}"] if under else [])
                    + [*map(str, cats)]
                    + ([f">{cats[-1]}"] if over else [])
                )
            case Regular() | Variable():
                _cats = []
                for a, b in cats[:-1]:
                    _cats.append(f"[{self._ff(a)},{self._ff(b)})")
                a, b = cats[-1]
                _cats.append(f"[{self._ff(a)},{self._ff(b)}{')' if over else ']'}")
                return (
                    ([f"(-\u221E,{self._ff(cats[0][0])})"] if under else [])
                    + _cats
                    + ([f"[{self._ff(cats[-1][-1])}, \u221E)"] if over else [])
                )

    def indexof(self, axis: HistAxis, bins: Iterable[str | int | bool]) -> list[int]:
        if isinstance(axis, (Regular, Variable)):
            return []
        under, _ = self.flow(axis)
        cats = [*axis]
        indices = []
        for b in bins:
            if (idx := cats.index(b)) >= 0:
                indices.append(idx + under)
        return indices

    def rebin(
        self, axis: HistAxis, rebin: int | list[int]
    ) -> tuple[HistAxis, npt.NDArray]:
        if (isinstance(rebin, int) and rebin <= 1) or (
            isinstance(rebin, list) and any(r < 1 for r in rebin)
        ):
            return axis, None
        if isinstance(rebin, int):
            _rebin = [*repeat(rebin, len(axis) // rebin)]
        else:
            _rebin = rebin.copy()
        if sum(_rebin) != len(axis):
            raise ValueError(f'Rebin "{rebin}" does not match axis "{axis}"')
        under, over = self.flow(axis)
        _rebin.insert(0, 0)
        _idx = np.cumsum(_rebin)
        _meta = dict(name=axis.name, label=axis.label)
        match axis:
            case Boolean():
                _axis = StrCategory(["All"], flow=False, **_meta)
            case IntCategory() | StrCategory():
                cats = [*axis]
                _axis = StrCategory(
                    [
                        "|".join(map(str, cats[_idx[i] : _idx[i + 1]]))
                        for i in range(len(_idx) - 1)
                    ],
                    flow=over,
                    **_meta,
                )
            case Integer():
                cats = [*axis]
                _axis = StrCategory(
                    (
                        ([f"<{cats[0]}"] if under else [])
                        + [
                            f"{cats[_idx[i]]}-{cats[_idx[i + 1]-1]}"
                            for i in range(len(_idx) - 1)
                        ]
                        + ([f">{cats[-1]}"] if over else [])
                    ),
                    flow=False,
                    **_meta,
                )
            case Regular() | Variable():
                edges = axis.edges
                _axis = Variable(
                    [edges[i] for i in _idx], underflow=under, overflow=over, **_meta
                )
        if under:
            _rebin.insert(1, 1)
        if over:
            _rebin.append(1)
        return _axis, np.cumsum(_rebin)[:-1]

    @overload
    def extend(self, axis: StrCategory, *values: str) -> StrCategory: ...
    @overload
    def extend(self, axis: IntCategory, *values: int) -> IntCategory: ...
    def extend(self, axis: HistAxis, *values):
        ax = axis.traits
        match axis:
            case IntCategory() | StrCategory():
                return type(axis)(
                    (*axis, *values),
                    name=axis.name,
                    label=axis.label,
                    growth=ax.growth,
                    overflow=ax.overflow,
                )
            case _:
                raise TypeError(f"Cannot extend <{axis.__class__.__name__}> axis")

    def equal(self, ax1: HistAxis, ax2: HistAxis):
        if type(ax1) is not type(ax2):
            return False
        if (self.flow(ax1), *ax1) != (self.flow(ax2), *ax2):
            return False
        return True
