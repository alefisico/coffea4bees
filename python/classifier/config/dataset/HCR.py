from __future__ import annotations

import re
from abc import abstractmethod
from functools import cache, cached_property
from typing import TYPE_CHECKING, Callable, Iterable

from classifier.df.tools import (
    add_columns,
    add_label_index,
    add_label_index_from_column,
    drop_columns,
    map_selection_to_index,
    prescale,
)
from classifier.task import ArgParser
from classifier.typetools import enum_dict
from classifier.utils import subgroups

from ..setting.df import Columns
from ..setting.HCR import Input, InputBranch, MassRegion, NTag
from ..setting.torch import KFold
from ._df import LoadGroupedRoot

if TYPE_CHECKING:
    import pandas as pd


class _Derived:
    region_index: str = "region_index"
    ntag_index: str = "ntag_index"


class _Common(LoadGroupedRoot):
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
                    "passHLT", Columns.event,
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

        friends = []
        for g in subs:
            if g in self.friends:
                friends.extend(self.friends[g])

        pres = []
        for gs, ps in self._preprocessor_by_label:
            if gs.intersection(subs):
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


def _FvT_apply_JCM(df):
    df.loc[df["threeTag"], "weight"] *= df.loc[df["threeTag"], "pseudoTagWeight"]
    return df


def _FvT_data_selection(df):
    return df[
        df["passHLT"]  # trigger
        & (df["SB"] | df["SR"])  # boson mass
        & (df["fourTag"] | df["threeTag"])  # n-tag
        & (~(df["SR"] & df["fourTag"]))  # blind
    ]


def _FvT_ttbar_selection(df):
    return df[
        df["passHLT"]  # trigger
        & (df["SB"] | df["SR"])  # boson mass
        & (df["fourTag"] | df["threeTag"])  # n-tag
    ]


def _FvT_ttbar_3b_prescale(df):
    return df["threeTag"]


class FvT(_Common):
    argparser = ArgParser()
    argparser.add_argument(
        "--ttbar-3b-prescale",
        default="10",
        help="prescale 3b ttbar events",
    )

    @property
    def allowed_labels(self):
        return []

    @property
    def preprocessor_by_label(self):
        return [
            (
                [("data",)],
                [
                    _FvT_data_selection,
                    add_label_index_from_column(threeTag="d3", fourTag="d4"),
                ],
            ),
            (
                [("ttbar",)],
                [
                    _FvT_ttbar_selection,
                    prescale(
                        scale=self.opts.ttbar_3b_prescale,
                        selection=_FvT_ttbar_3b_prescale,
                        seed=("ttbar", 0),
                    ),
                    add_label_index_from_column(threeTag="t3", fourTag="t4"),
                ],
            ),
            (
                [()],
                [
                    _FvT_apply_JCM,
                ],
            ),
        ]


class FvT_picoAOD(FvT):
    argparser = ArgParser()
    argparser.remove_argument("--files", "--filelists")
    defaults = {"files": [], "filelists": []}

    def __init__(self):
        super().__init__()
        self.opts.filelists = self._filelists()

    def _filelists(self):
        base = "metadata/datasets_HH4b_2024_v1.yml@@datasets.{dataset}.{year}.picoAOD{era}.files"
        year = {
            "2016": ["UL16_preVFP", "UL16_postVFP"],
            "2017": ["UL17"],
            "2018": ["UL18"],
        }
        era = {
            "UL16_preVFP": ["B", "C", "D", "E", "F"],
            "UL16_postVFP": ["F", "G", "H"],
            "UL17": ["B", "C", "D", "E", "F"],
            "UL18": ["A", "B", "C", "D"],
        }
        ttbar = ["TTTo2L2Nu", "TTToHadronic", "TTToSemiLeptonic"]
        filelists = []
        for k, v in year.items():
            files = [f"data,{k}"]
            for y in v:
                for e in era[y]:
                    files.append(base.format(dataset="data", year=y, era=f".{e}"))
            filelists.append(files)
        for k, v in year.items():
            files = [f"ttbar,{k}"]
            for y in v:
                for tt in ttbar:
                    files.append(base.format(dataset=tt, year=y, era=""))
            filelists.append(files)
        return filelists

    def debug(self):
        import logging

        from rich.pretty import pretty_repr

        logging.debug(pretty_repr(self.files))
