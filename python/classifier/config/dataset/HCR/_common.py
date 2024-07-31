from __future__ import annotations

import re
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import cache, cached_property
from typing import TYPE_CHECKING, Any, Callable, Iterable

from classifier.typetools import enum_dict

from ...setting.df import Columns
from ...setting.HCR import Input, InputBranch, MassRegion, NTag
from ...setting.torch import KFold
from .._df import LoadGroupedRoot

if TYPE_CHECKING:
    from classifier.df.tools import DFProcessor


class _Derived:
    region_index: str = "region_index"
    ntag_index: str = "ntag_index"


def _sort_map(obj: dict[frozenset[str]]):
    obj = {(*sorted(k),): v for k, v in obj.items()}
    return {k: obj[k] for k in sorted(obj)}


class _group_processor:
    def __init__(self, gs: Iterable[Iterable[str]], ps: Iterable[DFProcessor]):
        self._gs = (*map(frozenset, gs),)
        self._ps = (*ps,)

    def __call__(self, groups: frozenset[str]):
        if any(g <= groups for g in self._gs):
            yield from self._ps


@dataclass
class group_key:
    key: str = "year"
    pattern: str = r"year:\w*(?P<year>\d{2}).*"
    default: Any = None
    dtype: type = float

    def __post_init__(self):
        self._pattern = re.compile(self.pattern)

    def __call__(self, groups: frozenset[str]):
        from classifier.df.tools import add_columns

        matched = (*filter(None, map(self._pattern.fullmatch, groups)),)
        if len(matched) == 1:
            yield add_columns(**{self.key: self.dtype(matched[0].group(self.key))})
        elif len(matched) > 1:
            raise ValueError(f'Multiple "{self.pattern}" matched in {groups}')
        elif self.default is not None:
            yield add_columns(**{self.key: self.default})


class group_single_label:
    def __init__(self, *labels: str):
        self._labels = frozenset(labels)

    def __call__(self, groups: frozenset[str]):
        from classifier.df.tools import add_label_index

        matched = groups & self._labels
        if len(matched) == 1:
            yield add_label_index(next(iter(matched)))
        elif len(matched) > 1:
            raise ValueError(f"Multiple labels found in {groups}")


class Common(LoadGroupedRoot):
    def __init__(self):
        super().__init__()
        from classifier.df.tools import drop_columns, map_selection_to_flag

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
                map_selection_to_flag(
                    **enum_dict(MassRegion)
                ).set(name=_Derived.region_index),
                map_selection_to_flag(
                    **enum_dict(NTag)
                ).set(name=_Derived.ntag_index),
                drop_columns(
                    "ZZSR", "ZHSR", "HHSR", "SR", "SB",
                    "fourTag", "threeTag", "pseudoTagWeight",
                    "passHLT",
                ),
            ]
        )
        # fmt: on

    @cache
    def from_root(self, groups: frozenset[str]):
        from classifier.df.io import FromRoot

        friends = []
        for k, v in self.friends.items():
            if k <= groups:
                friends.extend(v)

        pres = []
        for g in self._preprocess_by_group:
            pres.extend(g(groups))
        pres.extend(self.preprocessors)

        return FromRoot(
            friends=friends,
            branches=self._branches.intersection,
            preprocessors=pres,
        )

    @cached_property
    def _preprocess_by_group(self) -> list[_group_processor]:
        pres = []
        for i in self.preprocess_by_group():
            if isinstance(i, tuple):
                pres.append(_group_processor(*i))
            else:
                pres.append(i)
        return pres

    @cached_property
    def _branches(self):
        return self.other_branches().union(
            InputBranch.feature_ancillary,
            InputBranch.feature_CanJet,
            InputBranch.feature_NotCanJet,
        )

    @abstractmethod
    def preprocess_by_group(
        self,
    ) -> Iterable[
        tuple[Iterable[Iterable[str]], Iterable[DFProcessor]]
        | Callable[[frozenset[str]], Iterable[DFProcessor]]
    ]: ...

    def other_branches(self):
        return {
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
        }

    def debug(self):
        import logging

        from rich.pretty import pretty_repr

        pres = defaultdict(list)
        for gs in self.files:
            for p in self._preprocess_by_group:
                pres[gs].extend(p(gs))
        logging.debug("files:", pretty_repr(_sort_map(self.files)))
        logging.debug("friends:", pretty_repr(_sort_map(self.friends)))
        logging.debug(
            "preprocessors:",
            pretty_repr(_sort_map(pres) | {"common": self.preprocessors}),
        )
        logging.debug("postprocessors:", pretty_repr(self.postprocessors))
        logging.debug("tensor:", pretty_repr(self.to_tensor._columns))
