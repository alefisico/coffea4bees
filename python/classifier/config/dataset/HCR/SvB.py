from __future__ import annotations

import operator as op
from functools import reduce
from typing import TYPE_CHECKING

from classifier.config.setting.df import Columns
from classifier.task import ArgParser

from . import _picoAOD
from ._common import Common, group_key, group_single_label

if TYPE_CHECKING:
    import pandas as pd


def _reweight_bkg(df: pd.DataFrame):
    df.loc[:, "weight"] *= df["pseudoTagWeight"] * df["FvT"]
    return df


class _common_selection:
    ntags: str

    def __init__(self, *regions: str):
        self.regions = regions

    def __call__(self, df: pd.DataFrame):
        return df[
            df["passHLT"]
            & df[self.ntags]
            & reduce(op.or_, (df[r] for r in self.regions))
        ]

    def __repr__(self):
        from classifier.df.tools import _iter_str, _type_str

        return f'{_type_str(self)} {_iter_str(("passHLT", self.ntags, *self.regions))}'


class _data_selection(_common_selection):
    ntags = "threeTag"


class _mc_selection(_common_selection):
    ntags = "fourTag"


class _Base(Common):
    argparser = ArgParser()
    argparser.add_argument(
        "--regions",
        nargs="+",
        default=["SR"],
        help="Dijet mass regions",
    )

    def __init__(self):
        super().__init__()
        self.to_tensor.add("kl", "float32").columns("kl")

    def preprocess_by_group(self):
        _MCs = ("ttbar", "ZZ", "ZH", "ggF", "VBF")
        return [
            group_key(),
            group_key("kl", r"kl:(?P<kl>.*)"),
            group_single_label("data", *_MCs),
            ([("data",)], [_reweight_bkg, _data_selection(*self.opts.regions)]),
            ([(k,) for k in _MCs], [_mc_selection(*self.opts.regions)]),
        ]


def _background_normalization(df: pd.DataFrame):
    df.loc[:, "weight"] /= df["weight"].sum()
    return df


class FvT(_picoAOD.Background, _Base):
    def __init__(self):
        super().__init__()
        self.postprocessors.append(_background_normalization)

    def other_branches(self):
        return super().other_branches() + {"FvT"}

    def preprocess_by_group(self):
        from classifier.df.tools import drop_columns

        return super().preprocess_by_group() + [
            ([()], [drop_columns("FvT")]),
        ]


def _signal_normalization(df: pd.DataFrame):
    group = df.groupby(
        [Columns.label_index, "kl"],
        dropna=False,
    )
    df.loc[:, "weight"] = group["weight"].transform(lambda x: x / x.sum())
    return df


class Signal(_picoAOD.Signal, _Base):
    def __init__(self):
        super().__init__()
        self.postprocessors.append(_signal_normalization)

    def other_branches(self):
        return super().other_branches() - {"pseudoTagWeight"}
