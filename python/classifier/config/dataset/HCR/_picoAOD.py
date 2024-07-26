from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property

from classifier.config.setting.cms import CollisionData, TTbarMC
from classifier.task import ArgParser, Dataset


class _PicoAOD(ABC, Dataset):
    argparser = ArgParser()
    argparser.remove_argument("--files", "--filelists")
    argparser.add_argument(
        "--metadata",
        default="datasets_HH4b",
        help="name of the metadata file",
    )
    defaults = {"files": [], "filelists": []}

    def __init__(self):
        super().__init__()
        self.opts.filelists = self._filelists()

    @cached_property
    def _metadata(self):
        return f"metadata/{self.opts.metadata}.yml@@datasets.{{dataset}}.{{year}}.picoAOD{{era}}.files"

    @abstractmethod
    def _filelists(self) -> list[list[str]]: ...


class FvT(_PicoAOD):
    def _filelists(self):
        filelists = []
        for year, eras in CollisionData.eras.items():
            filelists.append(
                [
                    f"data,year:{year}",
                    *(
                        self._metadata.format(dataset="data", year=year, era=f".{e}")
                        for e in eras
                    ),
                ]
            )
            filelists.append(
                [
                    f"ttbar,year:{year}",
                    *(
                        self._metadata.format(dataset=tt, year=year, era="")
                        for tt in TTbarMC.datasets
                    ),
                ]
            )
        return filelists

