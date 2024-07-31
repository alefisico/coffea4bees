from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.task import ArgParser

from . import _picoAOD
from ._common import Common, group_key

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


class FvT(_picoAOD.Background, Common):
    argparser = ArgParser()
    argparser.add_argument(
        "--ttbar-3b-prescale",
        default="10",
        help="prescale 3b ttbar events",
    )

    def preprocess_by_group(self):
        from classifier.df.tools import add_label_index_from_column, prescale

        return [
            (
                [("data",)],
                [
                    _data_selection,
                    add_label_index_from_column(threeTag="d3", fourTag="d4"),
                ],
            ),
            (
                [("ttbar",)],
                [
                    _ttbar_selection,
                    prescale(
                        scale=self.opts.ttbar_3b_prescale,
                        selection=_ttbar_3b_prescale,
                        seed=("ttbar", 0),
                    ),
                    add_label_index_from_column(threeTag="t3", fourTag="t4"),
                ],
            ),
            (
                [()],
                [
                    _apply_JCM,
                ],
            ),
            group_key(),
        ]
