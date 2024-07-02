from __future__ import annotations

import re
from abc import abstractmethod
from functools import cache, cached_property
from typing import TYPE_CHECKING, Callable, Iterable

from classifier.typetools import enum_dict

from ...setting.df import Columns
from ...setting.HCR import Input, InputBranch, MassRegion, NTag
from ...setting.torch import KFold
from .._df import LoadGroupedRoot

if TYPE_CHECKING:
    import pandas as pd


class _Derived:
    region_index: str = "region_index"
    ntag_index: str = "ntag_index"


class Common(LoadGroupedRoot):
    _year_pattern = re.compile(r"20\d{2}")

    _other_branches = [
        "ZZSR",
        "ZHSR",
        "HHSR",
        "SR",
        "SB",
        "fourTag",
        "threeTag",
        "passHLT",
        "pseudoTagWeight",
        Columns.event,
        Columns.weight,
    ]

    def __init__(self):
        super().__init__()
        from classifier.df.tools import drop_columns, map_selection_to_index

        # fmt: off
        (
            self.to_tensor
            .add(KFold.offset, KFold.offset_dtype).columns(Columns.event)
            .add(Input.label, Columns.index_dtype).columns(Columns.label_index)
            .add(Input.region, Columns.index_dtype).columns(_Derived.region_index)
            .add(Input.weight, "float32").columns(Columns.weight)
            .add(Input.ancillary, "float32").columns(*InputBranch.feature_ancillary)
            .add(Input.CanJet, "float32").columns(*InputBranch.feature_CanJet, target=InputBranch.n_CanJet)
            .add(Input.NotCanJet, "float32").columns(*InputBranch.feature_NotCanJet, target=InputBranch.n_NotCanJet)
        )
        self.preprocessors.extend(
            [
                map_selection_to_index(
                    **enum_dict(MassRegion)
                ).set(selection=_Derived.region_index, op="|"),
                map_selection_to_index(
                    **enum_dict(NTag)
                ).set(selection=_Derived.ntag_index),
                drop_columns(
                    "ZZSR", "ZHSR", "HHSR", "SR", "SB",
                    "fourTag", "threeTag", "pseudoTagWeight",
                    "passHLT",
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
        from classifier.df.tools import add_columns, add_label_index

        friends = []
        for k, v in self.friends.items():
            if k <= groups:
                friends.extend(v)

        pres = []
        for gs, ps in self._preprocessor_by_label:
            if any(map(lambda g: g <= groups, gs)):
                pres.extend(ps)
        pres.extend(self.preprocessors)

        year = self._get_year(groups)
        if year is not None:
            pres.append(add_columns(year=year))
        label = self._get_label(groups)
        if label is not None:
            pres.append(add_label_index(label))

        return FromRoot(
            friends=friends,
            branches=self._branches.intersection,
            preprocessors=pres,
        )

    @cached_property
    def _allowed_labels(self):
        return frozenset(self.allowed_labels)

    @cached_property
    def _preprocessor_by_label(self):
        return [
            ([frozenset(g) for g in gs], [*ps]) for gs, ps in self.preprocessor_by_label
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
