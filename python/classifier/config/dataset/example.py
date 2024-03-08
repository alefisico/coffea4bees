# TODO for test only, will be removed or rewritten
# TODO work in progress to match new skimmed data
from __future__ import annotations

import re
from abc import abstractmethod
from functools import cache
from typing import TYPE_CHECKING, Callable, Iterable

from classifier.df.tools import (
    add_event_offset,
    add_label_index_from_column,
    drop_columns,
    map_selection_to_index,
)

from ...utils import subgroups
from ..model.example import features
from ..setting.df import Columns
from ._df import LoadGroupedRoot

if TYPE_CHECKING:
    import pandas as pd


class _Common(LoadGroupedRoot):
    _year_pattern = re.compile(r"20\d{2}")

    _CanJet = [*map("CanJet_{}".format, features.candidate_jet)]
    _NotCanJet = [*map("NotCanJet_{}".format, features.other_jet)]
    _branches = {
        "ZZSR",
        "ZHSR",
        "HHSR",
        "SB",
        "fourTag",
        "threeTag",
        "passHLT",
        "nSelJets",
        "weight",
        "event",
        *_CanJet,
        *_NotCanJet,
    }

    def __init__(self):
        super().__init__()
        # fmt: off
        (
            self.to_tensor
            .add(Columns.event_offset, Columns.index_dtype).columns(Columns.event_offset)
            .add(Columns.label_index, Columns.index_dtype).columns(Columns.label_index)
            .add("region_index", Columns.index_dtype).columns("region_index")
            .add("weight", "float32").columns("genWeight")
            .add("ancillary", "float32").columns(*features.ancillary)
            .add("candidate_jet", "float32").columns(*self._CanJet, target=features.candidate_jet_max)
            .add("other_jet", "float32").columns(*self._NotCanJet, target=features.other_jet_max)
        )
        # fmt: on
        self.preprocessors.extend(
            [
                add_event_offset(60),  # 1, 2, 3, 4, 5, 6 folds
                map_selection_to_index(
                    SB=0b10, ZZSR=0b00101, ZHSR=0b01001, HHSR=0b10001
                ).set(selection="region_index"),
                map_selection_to_index(fourTag=0b10, threeTag=0b01).set(
                    selection="ntag_index"
                ),
                drop_columns(
                    "ZZSR",
                    "ZHSR",
                    "HHSR",
                    "SB",
                    "fourTag",
                    "threeTag",
                    "passHLT",
                    "event",
                ),
            ]
        )

    def _get_year(self, groups: frozenset[str]):
        matched = [*filter(self._year_pattern.fullmatch, groups)]
        if len(matched) >= 1:
            return int(matched[0])

    def _get_label(self, groups: frozenset[str]):
        matched = groups.intersection(self.allowed_labels)
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
        for gs, ps in self._preprocessor_by_label():
            if gs.intersection(groups):
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
            branches=self._branches,
            preprocessors=pres,
            metadata=metadata,
        )

    @classmethod
    @cache
    def _allowed_labels(cls):
        return frozenset(cls.allowed_labels())

    @classmethod
    @cache
    def _preprocessor_by_label(cls):
        return [
            (set(frozenset(g) for g in gs), [*ps])
            for gs, ps in cls.preprocessor_by_label()
        ]

    @classmethod
    @abstractmethod
    def allowed_labels(cls) -> Iterable[str]: ...

    @classmethod
    @abstractmethod
    def preprocessor_by_label(
        cls,
    ) -> Iterable[
        tuple[Iterable[Iterable[str]], Iterable[Callable[[pd.DataFrame], pd.DataFrame]]]
    ]: ...


class FvT(_Common):
    def __init__(self):
        super().__init__()
        # TODO normalization

    def allowed_labels(cls):
        return []

    def preprocessor_by_label(cls):
        return [
            (
                [
                    ("data",),
                ],
                [
                    add_label_index_from_column(threeTag="d3", fourTag="d4"),
                ],
            ),
            (
                [
                    ("ttbar",),
                ],
                [
                    add_label_index_from_column(threeTag="t3", fourTag="t4"),
                ],
            ),
        ]
