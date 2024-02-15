from abc import ABC, abstractmethod
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import UUID

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from ..monitor.dataflow import DfColumns, Source
from ..static import Setting
from ..utils import version_check

__all__ = [
    'Chunk',
    'Reader',
    'FromRoot',
    'FromFriendTree'
]

version_check(uproot, upper='5.0.0')
version_check(ak, upper='2.0.0')


class Chunk:
    def __init__(
            self,
            path: PathLike,  # TODO use EOS
            start: int,
            stop: int,
            uuid: UUID = None):
        self.path = path
        self.start = start
        self.stop = stop
        self.uuid = uuid

    def __repr__(self):
        return f'{self.path}[{self.uuid}]:{self.start}-{self.stop}'


class FlattenName:
    def __init__(self, name: str, n: int):
        name = name.split('_')
        name[0] = name[0] + '{index}'
        self.name = '_'.join(name)
        self.n = n

    def __iter__(self):
        for i in range(self.n):
            yield i, self.name.format(index=i)


class Reader(ABC):
    @abstractmethod
    def _read(
            self,
            path: PathLike,
            start: int = None,
            stop: int = None) -> pd.DataFrame:
        ...

    def read(
            self,
            path: PathLike,
            start: int = None,
            stop: int = None,
            selection: Callable[[pd.DataFrame], pd.DataFrame] = None,
            transform: Callable[[pd.DataFrame], pd.DataFrame] = None,
            **metadata) -> pd.DataFrame:
        df = self._read(path, start, stop)
        if selection is not None:
            df = selection(df)
        for k, v in metadata.items():
            df[k] = v
        if transform is not None:
            df = transform(df)
        return df

    @abstractmethod
    def _file_info(
            self,
            path: PathLike) -> tuple[int, UUID]:
        ...

    def iter(
            self,
            path: PathLike,
            chunk: int = None,
            return_proxy: bool = False,
            return_info: bool = False,
            selection: Callable[[pd.DataFrame], pd.DataFrame] = None,
            transform: Callable[[pd.DataFrame], pd.DataFrame] = None,
            **metadata):
        size, uuid = self._file_info(path)
        chunk = chunk or size
        for start in range(0, size, chunk):
            stop = min(start + chunk, size)
            if return_proxy:
                result = partial(self.read, path, start, stop,
                                 selection, transform, **metadata)
            else:
                result = self.read(path, start, stop,
                                   selection, transform, **metadata)
            if return_info:
                result = result, Chunk(path, start, stop, uuid)
            yield result


class FromRoot(Reader):
    def __init__(
            self,
            branches: list[str],
            flatten: list[tuple[str, Optional[int], Optional[Any]]] = None,
            tree: str = 'Events'):
        self.branches = branches
        self.flatten = [] if flatten is None else flatten.copy()
        self.tree = tree

    def add_flatten(
            self,
            branch: str,
            n: int = None,
            padding=None):
        self.flatten.append((branch, n, padding))
        return self

    def _read(
            self,
            path: PathLike,  # TODO use EOS
            start: int = None,
            stop: int = None) -> pd.DataFrame:
        if not (isinstance(path, str) and path.startswith('root://')):
            path = Path(path).resolve()
            if not path.exists():
                raise FileNotFoundError(path)

        with uproot.open(path)[self.tree] as tree:
            branches = set(tree.keys(filter_name=self.branches))
            flatten = {}
            for k, *pad in self.flatten:
                matched = tree.keys(filter_name=k)
                flatten |= dict.fromkeys(matched, pad)
            data: pd.DataFrame = tree.arrays(
                branches - set(flatten),
                library='pd',
                entry_start=start,
                entry_stop=stop,
            )
            jagged: ak.Array = tree.arrays(
                flatten.keys(),
                library='ak',
                entry_start=start,
                entry_stop=stop,
            )
            for k, (n, v) in flatten.items():
                branch = jagged[k]
                if n is None:
                    n = ak.max(ak.num(branch))
                branch = ak.pad_none(branch, n, clip=True, axis=1)
                for i, name in FlattenName(k, n):
                    column = branch[:, i].to_numpy(allow_missing=True)
                    if v is not None:
                        column = np.ma.filled(column, v)
                    data[name] = column
            if Setting.debug:
                for k in branches - set(flatten):
                    DfColumns.add_derived(k, Source('root'))
                for k, (n, _) in flatten.items():
                    for _, name in FlattenName(k, n):
                        DfColumns.add_derived(
                            name, Source('root'))
            return data

    def _file_info(self, path: PathLike) -> tuple[int, UUID]:
        with uproot.open(path)[self.tree] as tree:
            return tree.num_entries, tree.file.uuid


class FromFriendTree(FromRoot):
    ...  # TODO
