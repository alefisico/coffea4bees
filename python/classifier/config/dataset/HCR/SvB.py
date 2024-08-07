from __future__ import annotations

import operator as op
from functools import reduce
from typing import TYPE_CHECKING

from classifier.config.setting.df import Columns
from classifier.task import ArgParser

from . import _group, _picoAOD
from ._common import Common

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
        import numpy as np

        return [
            _group.regex(
                "label:data",
                (_data_selection(*self.opts.regions), _reweight_bkg),
                (_mc_selection(*self.opts.regions),),
            ),
            _group.add_year(),
            _group.add_column(
                key="kl", pattern=r"kl:(?P<kl>.*)", default=np.nan, dtype=float
            ),
            _group.add_single_label(),
        ]


class Background(_picoAOD.Background, _Base):
    def __init__(self):
        from classifier.df.tools import drop_columns

        super().__init__()
        self.postprocessors.append(self.normalize)
        self.preprocessors.append(drop_columns("FvT"))

    def other_branches(self):
        return super().other_branches() | {"FvT"}

    @staticmethod
    def normalize(df: pd.DataFrame):
        df.loc[:, "weight"] /= df["weight"].sum()
        return df


class Signal(_picoAOD.Signal, _Base):
    def __init__(self):
        super().__init__()
        self.postprocessors.append(self.normalize)

    def other_branches(self):
        return super().other_branches() - {"pseudoTagWeight"}

    @staticmethod
    def normalize(df: pd.DataFrame):
        group = df.groupby(
            [Columns.label_index, "kl"],
            dropna=False,
        )
        df.loc[:, "weight"] = group["weight"].transform(lambda x: x / x.sum())
        return df
