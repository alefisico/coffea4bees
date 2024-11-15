from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from functools import cache, cached_property, partial
from itertools import chain
from typing import Iterable

from classifier.task import ArgParser, parse
from classifier.typetools import enum_dict

from ...setting.df import Columns
from ...setting.HCR import Input, InputBranch, MassRegion, NTag
from ...setting.torch import KFold
from .._root import LoadGroupedRoot
from . import _group


class _Derived:
    region_index: str = "region_index"
    ntag_index: str = "ntag_index"


def _sort_map(obj: dict[frozenset[str]]):
    obj = {(*sorted(k),): v for k, v in obj.items()}
    return {k: obj[k] for k in sorted(obj)}


class Common(LoadGroupedRoot):
    argparser = ArgParser()
    argparser.add_argument(
        "--branch",
        metavar="BRANCH",
        nargs="+",
        action="extend",
        default=[],
        help="additional branches",
    )
    argparser.add_argument(
        "--preprocess",
        metavar=("GROUP", "PROCESSOR"),
        nargs="+",
        action="append",
        default=[],
        help="additional preprocessors",
    )

    @cache
    def from_root(self, groups: frozenset[str]):
        from classifier.df.io import FromRoot

        friends = []
        for k, v in self.friends.items():
            if k <= groups:
                friends.extend(v)

        pres = []
        for g in chain(self._preprocess_from_opts, self._preprocess_by_group):
            pres.extend(g(groups))
        pres.extend(self.preprocessors)

        return FromRoot(
            friends=friends,
            branches=self._branches.intersection,
            preprocessors=pres,
        )

    @cached_property
    def _preprocess_from_opts(self):
        return [
            _group.fullmatch(
                opts[0].split(","),
                processors=[partial(parse.instance, opts[1:], "classifier")],
            )
            for opts in self.opts.preprocess
        ]

    @cached_property
    def _preprocess_by_group(self):
        return self.preprocess_by_group()

    @abstractmethod
    def preprocess_by_group(self) -> Iterable[_group.ProcessorGenerator]: ...

    @cached_property
    def _branches(self):
        return self.other_branches().union(
            InputBranch.feature_ancillary,
            InputBranch.feature_CanJet,
            InputBranch.feature_NotCanJet,
            self.opts.branch,
        )

    @abstractmethod
    def other_branches(self) -> set[str]: ...


class CommonTrain(Common):
    trainable = True

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

        from classifier.config.state.label import MultiClass
        from rich.pretty import pretty_repr

        pres = defaultdict(list)
        for gs in self.files:
            for p in chain(self._preprocess_from_opts, self._preprocess_by_group):
                pres[gs].extend(p(gs))
        logging.debug(
            "friends:",
            pretty_repr(
                _sort_map(
                    {
                        k: [(v.name, v.branches) for v in vs]
                        for k, vs in self.friends.items()
                    }
                )
            ),
        )
        logging.debug("files:", pretty_repr(_sort_map(self.files)))
        logging.debug(
            "preprocessors:",
            pretty_repr(_sort_map(pres) | {"common": self.preprocessors}),
        )
        logging.debug("postprocessors:", pretty_repr(self.postprocessors))
        logging.debug("tensor:", pretty_repr(self.to_tensor._columns))
        logging.debug("labels:", pretty_repr(MultiClass.labels))


class CommonEval(Common):
    evaluable = True

    def __init__(self):
        super().__init__()

        # fmt: off
        (
            self.to_tensor
            .add(KFold.offset, KFold.offset_dtype).columns(Columns.event)
            .add(Input.ancillary, "float32").columns(*InputBranch.feature_ancillary)
            .add(Input.CanJet, "float32").columns(*InputBranch.feature_CanJet, target=InputBranch.n_CanJet)
            .add(Input.NotCanJet, "float32").columns(*InputBranch.feature_NotCanJet, target=InputBranch.n_NotCanJet)
        )
        # fmt: on

    def other_branches(self):
        return {
            Columns.event,
        }

    def preprocess_by_group(self):
        return [
            _group.add_year(),
        ]

    def debug(self):
        import logging

        from rich.pretty import pretty_repr

        logging.debug(
            "friends:",
            pretty_repr(
                _sort_map(
                    {
                        k: [(v.name, v.branches) for v in vs]
                        for k, vs in self.friends.items()
                    }
                )
            ),
        )
        logging.debug("files:", pretty_repr(_sort_map(self.files)))
        logging.debug("tensor:", pretty_repr(self.to_tensor._columns))
