from __future__ import annotations

import os
from functools import partial
from threading import Thread
from typing import TYPE_CHECKING, NotRequired, Optional, TypedDict

import cloudpickle
import lz4.frame
from src.config import ConfigParser, parsers
from src.storage.eos import EOS

from ._io import dumps

if TYPE_CHECKING:
    from io import BytesIO

    from dask.base import DaskMethodsMixin
    from dask.distributed import Client

    class DaskTasks(TypedDict):
        tasks: list[DaskMethodsMixin]
        "A collection of dask tasks."
        client: NotRequired[Client]
        "A dask distributed client."


class Outputs:
    _files = {}

    @classmethod
    def _parser(cls, key: str, value: os.PathLike, tag: Optional[str]):
        cls._files[os.fspath(value)] = tag
        return key, value

    @classmethod
    def _collect(
        cls,
        _,
        files: dict[str, Optional[str]],
        base: Optional[os.PathLike],
        clean: bool,
    ):
        base = EOS(base)
        if base.is_null:
            return
        kwargs = dict(parents=True, overwrite=True, recursive=True)
        op = EOS.move_to if clean else EOS.copy_to
        threads = list[Thread]()
        for src, dst in files.items():
            src = EOS(src)
            dst = dst or src.name
            threads.append(Thread(target=op, args=(src, base / dst), kwargs=kwargs))
            threads[-1].start()
        for t in threads:
            t.join()

    @classmethod
    def collect(cls, base: os.PathLike, clean: bool = False):
        """Collect all registered output files into the base directory."""
        return partial(cls._collect, files=cls._files.copy(), base=base, clean=clean)


@dumps.register("coffea")
def _coffea_serializer(file: BytesIO, obj):
    lz4file = lz4.frame.open(file, mode="wb")
    cloudpickle.dump(obj, lz4file)
    lz4file.flush()


@ConfigParser.io.register("coffea")
def _coffea_deserializer(file: BytesIO):
    lz4file = lz4.frame.open(file, mode="rb")
    data = cloudpickle.load(lz4file)
    return data


def load_configs(*configs: str) -> DaskTasks:
    "Load config files"
    return ConfigParser(
        tag_parsers={
            # <task>: a shortcut of <type> to import inside analysis_dask.
            "task": parsers.TypeParser("analysis_dask"),
            # <output>: register an output file.
            "output": Outputs._parser,
        }
    )(*configs)


def unique_client(address, **kwargs):
    "Check if there is a :class:`distributed.Client` running before creating a new one."
    from distributed import Client, get_client

    try:
        current = get_client()
    except ValueError:
        return Client(address, **kwargs)
    else:
        raise ValueError(f"There is already a client running: {current}") from None
