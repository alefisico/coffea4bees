from __future__ import annotations

import logging
from functools import cached_property
from inspect import getmro
from typing import Callable, Iterable

from classifier.config.setting.cms import CollisionData, MC_HH_ggF, MC_TTbar
from classifier.config.state import Flags
from classifier.task import ArgParser, Dataset, parse


class _PicoAOD(Dataset):
    pico_filelists: Iterable[Callable[[str], Iterable[list[str]]]]
    pico_files: Iterable[Callable[[str], Iterable[list[str]]]]

    argparser = ArgParser()
    argparser.remove_argument("--files", "--filelists")
    argparser.add_argument(
        "--metadata",
        nargs="*",
        default=["datasets_HH4b_2024_v2"],
        help="names of the metadata files.",
    )

    def __init__(self):
        super().__init__()
        if not hasattr(self.opts, "filelists"):
            self.opts.filelists = []
        if not hasattr(self.opts, "files"):
            self.opts.files = []
        for metadata in self.opts.metadata:
            self.opts.filelists.extend(
                self._filelists(f"metadata/{metadata}.yml@@datasets")
            )
        for metadata in self.opts.metadata:
            self.opts.files.extend(self._files(f"metadata/{metadata}.yml@@datasets"))

    def _iter(self, name: str):
        for base in getmro(self.__class__):
            if issubclass(base, _PicoAOD) and (
                (datasets := vars(base).get(name)) is not None
            ):
                yield from datasets

    def _load(self, name: str, metadata: str):
        filelists = []
        for dataset in self._iter(name):
            filelists.extend(dataset(self, metadata))
        return filelists

    def _files(self, metadata: str):
        return self._load("pico_files", metadata)

    def _filelists(self, metadata: str):
        return self._load("pico_filelists", metadata)


class _MCDataset:
    processes: tuple[str, ...]


class _ttbar(_MCDataset):
    processes = ("ttbar",)

    def __new__(cls, self: MC, metadata: str):
        filelists = []
        if "ttbar" in self.mc_processes:
            for year in CollisionData.eras:
                filelists.append(
                    [
                        f"label:ttbar,year:{year}",
                        *(
                            metadata + f".{tt}.{year}.picoAOD.files"
                            for tt in MC_TTbar.datasets
                        ),
                    ]
                )
        return filelists


class _ZZ_ZH(_MCDataset):
    processes = ("ZZ", "ZH")

    def __new__(cls, self: MC, metadata: str):
        filelists = []
        datasets = {}
        if "ZZ" in self.mc_processes:
            datasets["ZZ"] = ["ZZ4b"]
        if "ZH" in self.mc_processes:
            datasets["ZH"] = ["ZH4b", "ggZH4b"]
        for year in CollisionData.eras:
            for label, processes in datasets.items():
                filelists.append(
                    [
                        f"label:{label},year:{year}",
                        *(metadata + f".{d}.{year}.picoAOD.files" for d in processes),
                    ]
                )
        return filelists


class _ggF(_MCDataset):
    processes = ("ggF",)

    @classmethod
    def __c2str(cls, coupling: float):
        return f"{coupling:.6g}".replace(".", "p")

    @classmethod
    def __cs2label(cls, couplings: dict[str, float]):
        return ",".join(f"{k}:{v:.6g}" for k, v in couplings.items())

    def __new__(cls, self: MC, metadata: str):
        from base_class.physics.kappa_framework import Coupling

        filelists = []
        datasets = {}
        if "ggF" in self.mc_processes:
            datasets[("ggF", "GluGluToHHTo4B_cHHH{kl}")] = Coupling(kl=MC_HH_ggF.kl)
        for year in CollisionData.eras:
            for (label, pattern), couplings in datasets.items():
                for c in couplings:
                    process = pattern.format(
                        **{k: cls.__c2str(v) for k, v in c.items()}
                    )
                    filelists.append(
                        [
                            f"label:{label},year:{year},{cls.__cs2label(c)}",
                            metadata + f".{process}.{year}.picoAOD.files",
                        ]
                    )
        return filelists


def _data(self: Data, metadata: str):
    filelists = []
    if "detector" in self.data_sources:
        for year, eras in CollisionData.eras.items():
            filelists.append(
                [
                    f"label:data,year:{year},source:detector",
                    *(metadata + f".data.{year}.picoAOD.{e}.files" for e in eras),
                ]
            )
    return filelists


def _mixeddata(self: Data, metadata: str):
    files = []
    if "mixed" in self.data_sources:
        samples = parse.intervals(self.opts.data_mixed_samples)
        for year in CollisionData.years:
            templates: list[str] = parse.mapping(
                metadata + f".mixeddata.{year}.picoAOD.files_template", default="file"
            )
            urls = []
            for template in templates:
                template = template.replace("XXX", "{sample}").format
                for i in samples:
                    urls.append(template(sample=i))
            files.append(
                [
                    f"label:data,year:{year},source:mixed",
                    *urls,
                ]
            )
    return files


def _synthetic(self: Data, metadata: str):
    files = []
    if "synthetic" in self.data_sources:
        from base_class.system.eos import EOS

        samples = parse.intervals(self.opts.data_synthetic_samples)
        for year, eras in CollisionData.eras.items():
            templates: list[str] = parse.mapping(
                metadata + f".synthetic_data.{year}.picoAOD.files_template",
                default="file",
            )
            urls = []
            for template in templates:
                if "data" not in EOS(template).parent.name:
                    continue  # HACK: remove TTbar files, may need to change in the future
                template = template.replace("XXX", "{sample}").format
                for i in samples:
                    urls.append(template(sample=i))
            files.append(
                [
                    f"label:data,year:{year},source:synthetic",
                    *urls,
                ]
            )
    return files


class Data(_PicoAOD):
    pico_filelists = (_data,)
    pico_files = (_mixeddata, _synthetic)

    argparser = ArgParser()
    argparser.add_argument(
        "--data-source",
        metavar="SOURCE",
        default=["detector"],
        choices=("detector", "mixed", "synthetic"),
        help="choose the source of the data",
        nargs="*",
    )
    argparser.add_argument(
        "--data-mixed-samples",
        metavar="SAMPLE",
        action="extend",
        nargs="+",
        default=[],
        help="index of mixed samples",
    )
    argparser.add_argument(
        "--data-synthetic-samples",
        metavar="SAMPLE",
        action="extend",
        nargs="+",
        default=[],
        help="index of synthetic samples",
    )

    @cached_property
    def data_sources(self) -> set[str]:
        return {*self.opts.data_source}


class MC(_PicoAOD):
    argparser = ArgParser()
    argparser.add_argument(
        "--mc-processes",
        metavar="PROCESS",
        nargs="*",
        default=None,
        help="list of MC processes. If not specified, all processes are used",
    )

    @cached_property
    def mc_processes(self) -> set[str]:
        selected = self.mc_processes_all
        if self.opts.mc_processes is not None:
            selected = selected.intersection(self.opts.mc_processes)
        if Flags.debug:
            logging.debug(
                "The following MC processes are selected:",
                f"{sorted(selected)} of {sorted(self.mc_processes_all)}",
            )
        return selected

    @cached_property
    def mc_processes_all(self) -> set[str]:
        processes = set()
        for dataset in self._iter("pico_filelists"):
            if isinstance(dataset, type) and issubclass(dataset, _MCDataset):
                processes.update(dataset.processes)
        return processes


class Background(Data, MC):
    pico_filelists = (_ttbar,)


class Signal(MC):
    pico_filelists = (_ZZ_ZH, _ggF)
