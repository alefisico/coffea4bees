# QCDAE: setup dataset loading. ROOT file -> DataFrame -> Dataset(Tensor)
# see classifier.config.dataset.HCR._common for more details

# NOTE: Do NOT import large libraries (e.g. torch, numpy, pandas and modules that import them) at the top level of any file in `classifier.config`
# some tasks/functions e.g. help/auto-complete will import this file to get the definition but not run the code, importing large libraries will slow down the process

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Iterable

from classifier.task import ArgParser
from classifier.task.special import WorkInProgress

from ...setting.QCDAE import Input, InputBranch
from .._df import LoadGroupedRoot

# LoadGroupedRoot accept paths/filelists associated with a comma seperated groups
# e.g. "--files data,year:18 path/to/file1.root path/to/file2.root" where the group is ("data", "year:18")
# e.g. "--filelists ttbar,year:16 path/to/filelist1.json@key.to.array" where the group is ("ttbar", "year:16") and all paths in filelist1.json["key"]["to"]["array"] will be loaded
# an empty group is allowed by using --files "" path/to/file.root which will be parsed as ()

if TYPE_CHECKING:
    from classifier.df.tools import DFProcessor


class Toy(WorkInProgress, LoadGroupedRoot):
    argparser = ArgParser(prog="load the toy dataset")
    argparser.add_argument(
        "--some-arg",  # add arguments if needed
    )

    def __init__(self):
        super().__init__()
        # fmt: off
        (
            self.to_tensor
            .add(Input.Jet, "float32").columns(*InputBranch.feature_Jet, target=InputBranch.n_Jet)
            .add(Input.weight, "float32").columns(Input.weight)
        )
        # fmt: on
        # construct the Tensor from DataFrame by using the columns in the following way
        # tensor["Jet"][0] = [Jet_pt[0], ..., Jet_pt[3], ..., Jet_mass[0], ..., Jet_mass[3]]
        # where the DataFrame looks like
        # index Jet_pt    Jet_eta   Jet_phi   Jet_mass
        # 0     [0,1,2,3] [0,1,2,3] [0,1,2,3] [0,1,2,3]
        # 1     [0,1,2,3] [0,1,2,3] [0,1,2,3] [0,1,2,3]
        # 2     [0,1,2,3] [0,1,2,3] [0,1,2,3] [0,1,2,3]

    @cache  # may be called multiple times, slightly improve performance
    def from_root(  # the reading is finally called by _load_df_from_root in classifier.config.dataset._df
        self, groups: frozenset[str]
    ):  # groups passed from arguments, so that different reading strategies can be applied to different files
        from classifier.df.io import FromRoot

        preprocessors: Iterable[DFProcessor] = []
        if ... in groups:
            preprocessors.append(...)  # add preprocessors based on group if needed

        return FromRoot(  # an helper class to load ROOT files into DataFrame and apply some transformations
            branches={
                *InputBranch.feature_Jet,
                Input.weight,
            }.intersection,  # a filter on branch names
            preprocessors=preprocessors,  # add preprocessors if needed
        )
