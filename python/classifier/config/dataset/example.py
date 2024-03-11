from __future__ import annotations

import re
from abc import abstractmethod
from functools import cache, cached_property
from typing import TYPE_CHECKING, Callable, Iterable

from classifier.df.tools import (
    add_event_offset,
    add_label_index_from_column,
    drop_columns,
    map_selection_to_index,
    prescale,
)

from ...utils import subgroups
from ..setting.df import Columns
from ..setting.HCR import InputBranch
from ._df import LoadGroupedRoot

if TYPE_CHECKING:
    import pandas as pd


class _Common(LoadGroupedRoot):
    _year_pattern = re.compile(r"20\d{2}")

    _other_branches = [
        "ZZSR",
        "ZHSR",
        "HHSR",
        "SB",
        "fourTag",
        "threeTag",
        "passHLT",
        "event",
        "weight",
    ]

    def __init__(self):
        super().__init__()
        # fmt: off
        (
            self.to_tensor
            .add(Columns.event_offset, Columns.index_dtype).columns(Columns.event_offset)
            .add(Columns.label_index, Columns.index_dtype).columns(Columns.label_index)
            .add("region_index", Columns.index_dtype).columns("region_index")
            .add(Columns.weight, "float32").columns(Columns.weight)
            .add("ancillary", "float32").columns(*InputBranch.feature_ancillary)
            .add("CanJet", "float32").columns(*InputBranch.feature_CanJet, target=InputBranch.n_CanJet)
            .add("NotCanJet", "float32").columns(*InputBranch.feature_NotCanJet, target=InputBranch.n_NotCanJet)
        )
        self.preprocessors.extend(
            [
                add_event_offset(60),  # 1, 2, 3, 4, 5, 6 folds
                map_selection_to_index(
                    SB=0b10, ZZSR=0b00101, ZHSR=0b01001, HHSR=0b10001
                ).set(selection="region_index"),
                map_selection_to_index(
                    fourTag=0b10, threeTag=0b01
                ).set(selection="ntag_index"),
                drop_columns(
                    "ZZSR", "ZHSR", "HHSR", "SB",
                    "fourTag", "threeTag",
                    "passHLT", "event",
                ),
            ]
        )
        # fmt: on

    def _get_year(self, groups: frozenset[str]):
        matched = [*filter(self._year_pattern.fullmatch, groups)]
        if len(matched) >= 1:
            return int(matched[0])

    def _get_label(self, groups: frozenset[str]):
        matched = groups.intersection(self._allowed_labels)
        if len(matched) >= 1:
            return next(iter(matched))

    @cache
    def from_root(self, groups: frozenset[str]):
        from classifier.df.io import FromRoot

        subs = {*subgroups(groups)}

        friends = None
        for g in subs:
            if g in self.friends:
                friends = self.friends[g]
                break

        pres = []
        for gs, ps in self._preprocessor_by_label:
            if gs.intersection(subs):
                pres.extend(ps)
        pres.extend(self.preprocessors)

        metadata = {}
        year = self._get_year(groups)
        if year:
            metadata["year"] = year
        label = self._get_label(groups)
        if label:
            metadata["label"] = label

        return FromRoot(
            friends=friends,
            branches=self._branches.intersection,
            preprocessors=pres,
            metadata=metadata,
        )

    @cached_property
    def _allowed_labels(self):
        return frozenset(self.allowed_labels)

    @cached_property
    def _preprocessor_by_label(self):
        return [
            (set(frozenset(g) for g in gs), [*ps])
            for gs, ps in self.preprocessor_by_label
        ]

    @cached_property
    def _branches(self):
        return set(
            self._other_branches
            + InputBranch.feature_ancillary
            + InputBranch.feature_CanJet
            + InputBranch.feature_NotCanJet
        )

    @property
    @abstractmethod
    def allowed_labels(self) -> Iterable[str]: ...

    @property
    @abstractmethod
    def preprocessor_by_label(
        self,
    ) -> Iterable[
        tuple[Iterable[Iterable[str]], Iterable[Callable[[pd.DataFrame], pd.DataFrame]]]
    ]: ...


def _FvT_selection(df):
    return df[df["SB"] & df["passHLT"] & (df["fourTag"] | df["threeTag"])]


def _ttbar_3b_prescale(df):
    return df["threeTag"]


class FvT(_Common):
    def __init__(self):
        super().__init__()
        # TODO normalization

    @property
    def allowed_labels(self):
        return []

    @property
    def preprocessor_by_label(self):
        return [
            (
                [
                    ("data",),
                ],
                [
                    _FvT_selection,
                    add_label_index_from_column(threeTag="d3", fourTag="d4"),
                ],
            ),
            (
                [
                    ("ttbar",),
                ],
                [
                    _FvT_selection,
                    prescale(10, selection=_ttbar_3b_prescale),
                    add_label_index_from_column(threeTag="t3", fourTag="t4"),
                ],
            ),
        ]
