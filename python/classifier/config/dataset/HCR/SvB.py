from __future__ import annotations

import operator as op
from functools import partial, reduce
from typing import TYPE_CHECKING

from classifier.config.setting.df import Columns
from classifier.task import ArgParser, converter

from . import _group, _picoAOD
from ._common import CommonEval, CommonTrain

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


def _remove_outlier(df: pd.DataFrame):
    # TODO: This is a temporary solution triggered by the following two events:
    #
    # GluGluToHHTo4B_cHHH2p45
    # root://cmsxrootd.fnal.gov//store/mc/RunIISummer20UL16NanoAODv9/GluGluToHHTo4B_cHHH2p45_TuneCP5_PSWeights_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/2810000/13EBD534-FF20-C34B-BCC7-521EEB2FD396.root
    # event:298138, weight: 726.4/0.0131
    #
    # GluGluToHHTo4B_cHHH5
    # root://cmsxrootd.fnal.gov//store/mc/RunIISummer20UL17NanoAODv9/GluGluToHHTo4B_cHHH5_TuneCP5_PSWeights_13TeV-powheg-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1/30000/ABFD7D49-76A1-2A48-A523-6811C8C7FA01.root
    # event:269373, weight: 5996/0.0802
    return df.loc[df["weight"] < 1]


class _Train(CommonTrain):
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
                [
                    lambda: _data_selection(*self.opts.regions),
                    lambda: _reweight_bkg,
                ],
                [
                    lambda: _mc_selection(*self.opts.regions),
                ],
            ),
            _group.add_year(),
            _group.add_column(
                key="kl", pattern=r"kl:(?P<kl>.*)", default=np.nan, dtype=float
            ),
            _group.add_single_label(),
            _group.regex(
                r"kl:(2.45|5)",
                [
                    lambda: _remove_outlier,
                ],
            ),
        ]


class Background(_picoAOD.Background, _Train):
    argparser = ArgParser()
    argparser.add_argument(
        "--norm",
        default=1.0,
        type=converter.float_pos,
        help="normalization factor",
    )

    def __init__(self):
        from classifier.df.tools import drop_columns

        super().__init__()
        self.postprocessors.append(partial(self.normalize, norm=self.opts.norm))
        self.preprocessors.append(drop_columns("FvT"))

    def other_branches(self):
        return super().other_branches() | {"FvT"}

    @staticmethod
    def normalize(df: pd.DataFrame, norm: float):
        df.loc[:, "weight"] /= df["weight"].sum() / norm
        return df


class Signal(_picoAOD.Signal_ZZZH, _picoAOD.Signal_ggF, _Train):
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


class Eval(
    _picoAOD.Background,
    _picoAOD.Signal_ZZZH,
    _picoAOD.Signal_ggF,
    CommonEval,
): ...
