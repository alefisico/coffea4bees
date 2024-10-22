from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.task import ArgParser

from . import _group, _picoAOD
from ._common import CommonEval, CommonTrain

if TYPE_CHECKING:
    import pandas as pd


def _apply_JCM(df: pd.DataFrame):
    df.loc[df["threeTag"], "weight"] *= df.loc[df["threeTag"], "pseudoTagWeight"]
    return df


def _common_selection(df: pd.DataFrame):
    return df["passHLT"] & (df["SB"] | df["SR"]) & (df["fourTag"] | df["threeTag"])


def _data_selection(df: pd.DataFrame):
    return df[_common_selection(df) & (~(df["SR"] & df["fourTag"]))]


def _ttbar_selection(df: pd.DataFrame):
    return df[_common_selection(df)]


def _ttbar_3b_prescale(df: pd.DataFrame):
    return df["threeTag"]


class Train(_picoAOD.Background, CommonTrain):
    argparser = ArgParser()
    argparser.add_argument(
        "--ttbar-3b-prescale",
        default="10",
        help="prescale 3b ttbar events",
    )

    def preprocess_by_group(self):
        from classifier.df.tools import add_label_index_from_column, prescale

        return [
            _group.fullmatch(
                ("label:data",),
                processors=[
                    _data_selection,
                    add_label_index_from_column(threeTag="d3", fourTag="d4"),
                ],
            ),
            _group.fullmatch(
                ("label:ttbar",),
                processors=[
                    prescale(
                        scale=self.opts.ttbar_3b_prescale,
                        selection=_ttbar_3b_prescale,
                        seed=("ttbar", 0),
                    ),
                    _ttbar_selection,
                    add_label_index_from_column(threeTag="t3", fourTag="t4"),
                ],
            ),
            _group.fullmatch(
                (),
                processors=[_apply_JCM],
            ),
            _group.add_year(),
        ]


class Eval(_picoAOD.Background, CommonEval): ...
