from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.df.tools import add_label_index_from_column, prescale
from classifier.task import ArgParser

from ._common import Common

if TYPE_CHECKING:
    import pandas as pd


def _apply_JCM(df: pd.DataFrame):
    df.loc[df["threeTag"], "weight"] *= df.loc[df["threeTag"], "pseudoTagWeight"]
    return df


def _data_selection(df: pd.DataFrame):
    return df[
        df["passHLT"]  # trigger
        & (df["SB"] | df["SR"])  # boson mass
        & (df["fourTag"] | df["threeTag"])  # n-tag
        & (~(df["SR"] & df["fourTag"]))  # blind
    ]


def _ttbar_selection(df: pd.DataFrame):
    return df[
        df["passHLT"]  # trigger
        & (df["SB"] | df["SR"])  # boson mass
        & (df["fourTag"] | df["threeTag"])  # n-tag
    ]


def _ttbar_3b_prescale(df: pd.DataFrame):
    return df["threeTag"]


class FvT(Common):
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
