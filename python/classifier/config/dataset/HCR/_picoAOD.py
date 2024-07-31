from __future__ import annotations

from abc import ABC, abstractmethod

from classifier.config.setting.cms import CollisionData, MC_HH_ggF, MC_TTbar
from classifier.task import ArgParser, Dataset


class _PicoAOD(ABC, Dataset):
    argparser = ArgParser()
    argparser.remove_argument("--files", "--filelists")
    argparser.add_argument(
        "--metadata",
        nargs="*",
        default=["datasets_HH4b"],
        help="names of the metadata files.",
    )

    def __init__(self):
        super().__init__()
        if not hasattr(self.opts, "filelists"):
            self.opts.filelists = []
        for metadata in self.opts.metadata:
            self.opts.filelists.extend(
                self._filelists(
                    f"metadata/{metadata}.yml@@datasets.{{dataset}}.{{year}}.picoAOD{{era}}.files"
                )
            )

    @abstractmethod
    def _filelists(self, metadata: str) -> list[list[str]]: ...


def _ttbar(metadata: str):
    filelists = []
    for year in CollisionData.eras:
        filelists.append(
            [
                f"ttbar,year:{year}",
                *(
                    metadata.format(dataset=tt, year=year, era="")
                    for tt in MC_TTbar.datasets
                ),
            ]
        )
    return filelists


def _data(metadata: str):
    filelists = []
    for year, eras in CollisionData.eras.items():
        filelists.append(
            [
                f"data,year:{year}",
                *(
                    metadata.format(dataset="data", year=year, era=f".{e}")
                    for e in eras
                ),
            ]
        )
    return filelists


def _ZZ_ZH(metadata: str):
    filelists = []
    datasets = {
        "ZZ": ["ZZ4b"],
        "ZH": ["ZH4b", "ggZH4b"],
    }
    for year in CollisionData.eras:
        for label, processes in datasets.items():
            filelists.append(
                [
                    f"{label},year:{year}",
                    *(metadata.format(dataset=d, year=year, era="") for d in processes),
                ]
            )
    return filelists


class _ggF:
    @classmethod
    def __c2str(cls, coupling: float):
        return f"{coupling:.6g}".replace(".", "p")

    @classmethod
    def __cs2label(cls, couplings: dict[str, float]):
        return ",".join(f"{k}:{v:.6g}" for k, v in couplings.items())

    def __new__(cls, metadata: str):
        from base_class.physics.kappa_framework import Coupling

        filelists = []
        datasets = {
            ("ggF", "GluGluToHHTo4B_cHHH{kl}"): Coupling(kl=MC_HH_ggF.kl),
        }
        for year in CollisionData.eras:
            for (label, process), couplings in datasets.items():
                for c in couplings:
                    filelists.append(
                        [
                            f"{label},year:{year},{cls.__cs2label(c)}",
                            metadata.format(
                                dataset=process.format(
                                    **{k: cls.__c2str(v) for k, v in c.items()}
                                ),
                                year=year,
                                era="",
                            ),
                        ]
                    )
        return filelists


class Background(_PicoAOD):
    def _filelists(self, metadata: str):
        return sum(map(lambda func: func(metadata), [_ttbar, _data]), [])


class Signal(_PicoAOD):
    def _filelists(self, metadata: str):
        return sum(map(lambda func: func(metadata), [_ZZ_ZH, _ggF]), [])
