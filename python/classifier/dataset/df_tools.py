import operator as op
from copy import deepcopy
from functools import partial, reduce
from os import PathLike
from typing import Callable, Literal, Optional

import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from ..monitor import console
from ..monitor.dataflow import DfColumns, Source, Target
from ..static import Constant, Setting
from .label import Label, LabelCollection
from .df_reader import Reader


__all__ = [
    'Constructor',
    'Normalizer',
    'ToTensorDataset',
    'add_unscaled_weight',
    'add_region_index',
    'add_event_offset',
]


def _call(func):
    return func()


class Constructor:
    def __init__(self, reader: Reader):
        self.reader = reader
        self.metadata: set[str] = None
        self._labels: type[LabelCollection] = None
        self._sources: dict[Label, list[tuple[
            PathLike,
            dict[str],
            Callable[[pd.DataFrame], pd.DataFrame],
            Callable[[pd.DataFrame], pd.DataFrame]
        ]]] = {None: []}

    def add(
            self,
            *paths: PathLike,
            label: Optional[Label] = None,
            selection: Callable[[pd.DataFrame], pd.DataFrame] = None,
            transform: Callable[[pd.DataFrame], pd.DataFrame] = None,
            **metadata):
        if label is not None:
            if self._labels is None:
                self._labels = label.base
                fields = [label.field for label in self._labels.all]
                if Setting.debug:
                    for field in fields:
                        DfColumns.add_derived(field, Source('label'))
                    DfColumns.add_derived(Constant.label_index, *fields)
            elif self._labels is not label.base:
                raise ValueError(
                    f'Label must be from {self._labels}, got {label} from {label.base}')
        if self.metadata is None:
            self.metadata = set(metadata)
            if Setting.debug:
                for column in self.metadata:
                    DfColumns.add_derived(
                        column, Source('metadata'))
        else:
            metanames = set(metadata)
            if self.metadata > metanames:
                raise ValueError(
                    f'Expected metadata {self.metadata - metanames}')
            elif metanames > self.metadata:
                console.warn(
                    f'The following metadata will be ignored: {metanames - self.metadata}')

        metadata = {k: metadata[k] for k in self.metadata}
        if label is not None:
            metadata |= label.label
            metadata[Constant.label_index] = Constant.index_dtype(
                label.index[0])

        self._sources.setdefault(label, []).extend(
            (path, metadata, selection, transform) for path in paths)

    def concat(
            self,
            *labels: Label,
            chunk: int = None) -> pd.DataFrame:
        return pd.concat(
            self.iter(*labels, chunk=chunk),
            ignore_index=True
        )

    def concat_parallel(
            self,
            *labels: Label,
            chunk: int = None,
            n_workers: int = None,
            mode: Literal['spawn', 'forkserver', 'thread'] = 'forkserver') -> pd.DataFrame:
        if n_workers is None:
            from multiprocessing import cpu_count
            n_workers = cpu_count()

        match mode:
            case 'spawn' | 'forkserver':
                from ..process import get_context
                ctx = get_context(mode)
                Pool = ctx.Pool
            case 'thread':
                from multiprocessing.pool import ThreadPool as Pool
            case _:
                raise ValueError(f'Unknown mode {mode}')

        with Pool(n_workers) as pool:
            return pd.concat(
                pool.map(
                    _call,
                    self.iter(*labels, chunk=chunk, return_proxy=True)),
                ignore_index=True
            )

    def iter(
            self,
            *labels: Label,
            chunk: int = None,
            return_proxy: bool = False,
            return_info: bool = False):
        if not labels:
            labels = set(self._sources.keys()) - {None}
        for label in labels:
            for path, meta, selection, transform in self._sources[label]:
                yield from self.reader.iter(
                    path,
                    chunk=chunk,
                    return_proxy=return_proxy,
                    return_info=return_info,
                    selection=selection,
                    transform=transform,
                    **meta
                )


class Normalizer:
    def __init__(self, target: str):
        self._target = target
        self._mask: list[Callable[[pd.DataFrame], pd.DataFrame]] = []
        self._group: list[str] = []

    def copy(self):
        new = Normalizer(self._target)
        new._mask = self._mask.copy()
        new._group = self._group.copy()
        return new

    @classmethod
    @property
    def _class(cls):
        return Target(cls.__name__)

    def any(self, *columns: str):
        if Setting.debug:
            DfColumns.add_derived(self._class, *columns)
        return self.select(lambda df: df[list(columns)].any(axis=1))

    def all(self, *columns: str):
        if Setting.debug:
            DfColumns.add_derived(self._class, *columns)
        return self.select(lambda df: df[list(columns)].all(axis=1))

    def groupby(self, *columns: str):
        if Setting.debug:
            DfColumns.add_derived(self._class, *columns)
        new = self.copy()
        new._group.extend(columns)
        return new

    def select(self, selected: Callable[[pd.DataFrame], pd.Series]):
        new = self.copy()
        new._mask.append(selected)
        return new

    def to(self, data: pd.DataFrame, value: float):
        total, masked, mask = self._norm(data)
        total = total / value
        if self._group:
            if len(self._group) == 1:
                keys = masked[self._group[0]]
            else:
                keys = pd.MultiIndex.from_frame(masked[self._group])
            total = keys.map(total)
        data.loc[mask, self._target] /= total
        return data

    def _norm(self, data: pd.DataFrame):
        columns = [self._target] + self._group
        mask = self.mask(data)
        masked = data.loc[mask, columns]
        if self._group:
            total = masked.groupby(self._group)[self._target].sum()
        else:
            total = masked[self._target].sum()
        return total, masked, mask

    def norm(self, data: pd.DataFrame):
        return self._norm(data)[0]

    def mask(self, data: pd.DataFrame):
        if not self._mask:
            return slice(None)
        return reduce(op.and_, (m(data) for m in self._mask))


class ToTensorDataset:
    def __init__(self):
        self.columns: list[tuple[str, list[str], npt.DTypeLike]] = []

    def copy(self):
        return deepcopy(self)

    def add_column(
            self,
            *columns: str,
            name: str = ...,
            dtype: npt.DTypeLike = None):
        if columns:
            if name is ...:
                name = columns[0]
            self.columns.append((name, [*columns], dtype))
            if Setting.debug:
                DfColumns.add_derived(Target('tensor'), *columns)
        return self

    def tensor(
            self,
            data: pd.DataFrame) -> tuple[list[str], TensorDataset]:
        dataset, columns = [], []
        for name, column, dtype in self.columns:
            if any(c not in data for c in column):
                continue
            columns.append(name)
            dataset.append(torch.from_numpy(
                data.loc[:, column].to_numpy(dtype)))
        return columns, TensorDataset(*dataset)


def add_unscaled_weight(
        weight: str = Constant.weight,
        df: pd.DataFrame = None):
    if Setting.debug:
        DfColumns.add_derived(Constant.unscaled_weight, weight)
    if df is None:
        return partial(add_unscaled_weight, weight)
    df[Constant.unscaled_weight] = df[weight]
    return df


def add_region_index(
        regions: list[str] | dict[str, int],
        df: pd.DataFrame = None):
    if isinstance(regions, list):
        regions = {region: i for i, region in enumerate(regions)}
    else:
        if len(regions) != len(set(regions.values())):
            raise ValueError('region indices must be unique')
        regions = regions.copy()
    if Setting.debug:
        DfColumns.add_derived(Constant.region_index, *regions)
    if df is None:
        return partial(add_region_index, regions)
    df[Constant.region_index] = reduce(
        op.add, (df[k] * Constant.index_dtype(v) for k, v in regions.items()))
    return df


def add_event_offset(
        kfold: int,
        index: str = Constant.event,
        df: pd.DataFrame = None):
    if Setting.debug:
        DfColumns.add_derived(Constant.event_offset, index)
    if df is None:
        return partial(add_event_offset, kfold, index)
    df[Constant.event_offset] = (
        df[index] % kfold).astype(Constant.index_dtype)
    return df
