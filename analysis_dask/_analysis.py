from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from os import PathLike
from typing import TYPE_CHECKING, TypedDict

from coffea.dataset_tools import apply_to_dataset, preprocess
from coffea.nanoevents.schemas import BaseSchema, NanoAODSchema
from coffea.processor import ProcessorABC

import src.dask.awkward as dak_ext
from src.config._io import FileLoader
from src.data_formats.root import Chunk, Friend
from src.storage.eos import EOS

from ._io import _FileDumper, id_sha1

if TYPE_CHECKING:
    import awkward as ak
    import dask_awkward as dak
    from coffea.dataset_tools.apply_processor import (
        DaskOutputType,
        FilesetSpec,
        GenericHEPAnalysis,
    )

    class AnalysisOutput(TypedDict):
        result: dict[str, DaskOutputType]
        report: dict[str, dak.Array]


_cache_loader = FileLoader(cache=False)
_cache_dumper = _FileDumper()


@dak_ext.partition_mapping(label="attach-friend-trees")
def _attach_friends(events, friends: dict[str, ak.Array]):
    for k, v in friends.items():
        events[k] = v
    return events


@dataclass
class _FriendTreeProcessor:
    processor: ProcessorABC | GenericHEPAnalysis
    friends: dict[str, Friend]
    chunks: list[Chunk]

    def __post_init__(self):
        if isinstance(self.processor, ProcessorABC):
            self._func = self.processor.process
        else:
            self._func = self.processor

    def __call__(self, events: dak.Array):
        friends = {}
        for k, v in self.friends.items():
            arr = v.dask(*self.chunks, library="ak")
            if arr is not None:
                friends[k] = arr
        return self._func(_attach_friends(events, friends))


def _repartition_dataset(dataset: FilesetSpec, step_size: int) -> FilesetSpec:
    for spec in dataset.values():
        for file in spec["files"].values():
            steps = file["steps"]
            file["steps"] = [
                [c.entry_start, c.entry_stop]
                for c in Chunk.balance(
                    step_size or math.inf,
                    Chunk(
                        num_entries=steps[-1][-1],
                        entry_start=steps[0][0],
                        entry_stop=steps[-1][-1],
                    ),
                )
            ]


def apply(
    processor: GenericHEPAnalysis | ProcessorABC,
    fileset: FilesetSpec,
    fileset_cache: PathLike = "/tmp/fileset-cache/",
    update_fileset_cache: bool = False,
    friend_trees: dict[str, dict | Friend] = None,
    step_size: int = None,
    schemaclass: type[BaseSchema] = NanoAODSchema,
    preprocess_options: dict = None,
    uproot_options: dict = None,
):
    """
    Apply processor to fileset. Basically a wrapper of the following functions with additional friend tree and cache support.

    - :func:`coffea.dataset_tools.preprocess`
    - :func:`coffea.dataset_tools.apply_to_dataset`

    Parameters
    ----------
        processor: ProcessorABC or GenericHEPAnalysis
            A callable compatible with :class:`dask_awkward.Array`.
        fileset: dict[str, DatasetSpec]
            A dictionary of datasets.
        fileset_cache: PathLike or None, default ``"/tmp/fileset-cache/"``
            A path to cache preprocessed fileset. If None, the cache will be disabled.
        update_fileset_cache: bool, optional
            If True, it will always ignore the cache and rerun the preprocess.
        friend_trees: dict[str, dict or Friend], optional
            A dictionary of friend trees to be attached to the main event tree before processing, where the keys will be used as the field names.
            A value can be a :class:`base_class.root.Friend` object or a dictionary that can be parsed by :meth:`base_class.root.Friend.from_json`.
        step_size: int, optional
            The target partition size passed to :func:`coffea.dataset_tools.preprocess`.
        schemaclass: type[BaseSchema], default :class:`coffea.nanoevents.schemas.NanoAODSchema`
            The schema class passed to :func:`coffea.dataset_tools.preprocess`.
        preprocess_options: dict, optional
            Additional options passed to :func:`coffea.dataset_tools.preprocess`.
        uproot_options: dict, optional
            Additional options passed to :func:`uproot.dask`.

    Returns
    -------
    AnalysisOutput
        A dictionary with two keys: `result` and `report`.
    """
    # preprocess
    cache = None
    others, files = {}, {}
    for name, dataset in sorted(fileset.items()):
        dataset = dataset.copy()
        files[name] = {"files": dataset.pop("files")}
        others[name] = dataset
    if fileset_cache is not None:
        fileset_cache = EOS(fileset_cache) / f"{id_sha1(files)}.json.lz4"
        if not update_fileset_cache:
            try:
                cache = _cache_loader.load(fileset_cache)
                if set(files.keys()) != set(cache["files"].keys()):
                    logging.warning(
                        f"The cached fileset is modified or you probably encountered a sha-1 collision. Cache file: {fileset_cache}"
                    )
                    raise FileNotFoundError
                available = cache["available"]
                logging.info(f"Using cached fileset from: {fileset_cache}")
            except FileNotFoundError:
                cache = None
    if cache is None:
        preprocess_options = (preprocess_options or {}) | {
            "save_form": True,
            "step_size": step_size,
            "fileset": files,
        }
        available, updated = preprocess(**preprocess_options)
        if fileset_cache is not None:
            _cache_dumper(
                {
                    "files": files,
                    "updated": updated,
                    "available": available,
                    "step_size": step_size,
                },
                fileset_cache,
            )
    elif step_size != cache["step_size"]:
        _repartition_dataset(available, step_size)

    # attach friend tree
    chunks = Chunk.from_coffea_datasets(available)
    friends = {}
    for name, friend in (friend_trees or {}).items():
        if isinstance(friend, dict):
            friend = Friend.from_json(friend)
        if isinstance(friend, Friend):
            friends[name] = friend
        else:
            raise ValueError(f"Invalid friend tree: {friend}")

    # modified apply_to_fileset
    result = {}
    report = {}
    for name, dataset in available.items():
        dataset = dataset | others[name]
        metadata = copy.deepcopy(dataset.get("metadata")) or {}
        metadata.setdefault("dataset", name)
        dataset_out = apply_to_dataset(
            data_manipulation=_FriendTreeProcessor(processor, friends, chunks[name]),
            dataset=dataset,
            schemaclass=schemaclass,
            metadata=metadata,
            uproot_options=uproot_options or {},
        )
        result[name] = dataset_out[0]
        if len(dataset_out) > 1:
            report[name] = dataset_out[1]
    return {"result": result, "report": report}
