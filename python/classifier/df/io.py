from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from base_class.root import Chain, Chunk, Friend


class FromRoot:
    def __init__(
        self,
        friends: list[Friend] = None,
        branches: Callable[[set[str]], set[str]] = None,
        selection: Callable[[pd.DataFrame], pd.DataFrame] = None,
        metadata: dict[str, Any] = None,
    ):
        self.chain = Chain()
        self.branches = branches
        self.selection = selection
        self.metadata = metadata or {}

        for friend in friends or ():
            self.chain.add_friend(friend, renaming='{friend}_{branch}')

    def read(self, chunk: Chunk):
        chain = self.chain.copy()
        chain += chunk
        df = None
        for df in chain.iterate(
            library='pd',
            reader_options={'filter': self.branches}
        ):
            if self.selection:
                df = self.selection(df)
        for k, v in self.metadata.items():
            df[k] = v
        return df


class ToTensor:
    def __init__(self):
        self._columns: dict[str, tuple[
            npt.DTypeLike, list[tuple[str, Optional[int], Any, Optional[tuple[int, ...]]]]]] = {}
        self._current: str = None

    def remove(self, name: str):
        self._columns.pop(name, None)
        return self

    def add(
        self,
        name: str,
        dtype: npt.DTypeLike = None,
    ):
        self._columns.setdefault(name, (dtype, []))
        self._current = name
        return self

    def columns(
        self,
        *columns: str,
        target: int = None,
        padding: Any = 0,
        reshape: tuple[int, ...] = None,
    ):
        if self._current is None:
            raise RuntimeError(
                'Call add() to specify a name before adding columns')
        self._columns[self._current][1].extend(
            (c, target, padding, reshape) for c in columns)
        return self

    def tensor(self, data: pd.DataFrame):
        dataset = {}
        for name, (dtype, columns) in self._columns.items():
            missing = [c for c, *_ in columns if c not in data]
            if missing:
                logging.warning(
                    f'columns {missing} not found in dataframe')
                continue
            arrays = []
            for c, target, padding, reshape in columns:
                if data[c].dtype == 'awkward':
                    if target is None:
                        target = int(np.max(data[c].ak.num()))
                    array = np.ma.filled(
                        data[c].ak.pad_none(target, clip=True).ak.to_numpy(), padding).astype(dtype)
                else:
                    array = data[c].to_numpy(dtype)
                    if len(array.shape) == 1 and len(columns) > 1:
                        array = array[:, None]
                if reshape is not None:
                    array = array.reshape(reshape)
                arrays.append(array)
            if len(arrays) == 1:
                to_tensor = arrays[0]
            else:
                to_tensor = np.concatenate(arrays, axis=1)
            dataset[name] = torch.from_numpy(to_tensor)
        return dataset
