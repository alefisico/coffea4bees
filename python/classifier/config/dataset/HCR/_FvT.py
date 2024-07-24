from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.task import ArgParser, Dataset

from ._common import Common

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


class FvT(Common):
    argparser = ArgParser()
    argparser.add_argument(
        "--ttbar-3b-prescale",
        default="10",
        help="prescale 3b ttbar events",
    )

    def label_from_group(self):
        return ()

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
        ]


class _PicoAOD(Dataset):
    argparser = ArgParser()
    argparser.remove_argument("--files", "--filelists")
    argparser.add_argument(
        "--metadata",
        default="datasets_HH4b_2024_v1",
        help="name of the metadata file",
    )
    defaults = {"files": [], "filelists": []}

    def __init__(self):
        super().__init__()
        self.opts.filelists = self._filelists()

    def _filelists(self):
        base = f"metadata/{self.opts.metadata}.yml@@datasets.{{dataset}}.{{year}}.picoAOD{{era}}.files"
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


class FvT_picoAOD(_PicoAOD, FvT): ...
