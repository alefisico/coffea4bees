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


def _ttbar_3b_selection(df: pd.DataFrame):
    return df["threeTag"]


def _ttbar_4b_selection(df: pd.DataFrame):
    return df["fourTag"]


class Train(_picoAOD.Background, CommonTrain):
    argparser = ArgParser()
    argparser.add_argument(
        "--no-JCM",
        action="store_true",
        help="disable JCM weights",
    )
    argparser.add_argument(
        "--no-ttbar-3b",
        action="store_true",
        help="remove 3b ttbar events",
    )
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
                name="data selection",
            ),
            (
                _group.fullmatch(
                    ("label:ttbar",),
                    processors=[
                        prescale(
                            scale=self.opts.ttbar_3b_prescale,
                            selection=_ttbar_3b_selection,
                            seed=("ttbar", 0),
                        ),
                        _ttbar_selection,
                        add_label_index_from_column(threeTag="t3", fourTag="t4"),
                    ],
                    name="ttbar selection",
                )
                if not self.opts.no_ttbar_3b
                else _group.fullmatch(
                    ("label:ttbar",),
                    processors=[
                        _ttbar_4b_selection,
                        _ttbar_selection,
                        add_label_index_from_column(fourTag="t4"),
                    ],
                    name="ttbar 4b selection",
                )
            ),
            *(
                (
                    _group.fullmatch(
                        (),
                        processors=[_apply_JCM],
                        name="apply JCM",
                    ),
                )
                if not self.opts.no_JCM
                else ()
            ),
            _group.add_year(),
        ]


class Eval(_picoAOD.Background, CommonEval): ...
