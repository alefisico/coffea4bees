from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from base_class.root import Chain, Chunk, Friend


class FromRoot:
    def __init__(
        self,
        friends: Iterable[Friend] = None,
        branches: Callable[[set[str]], set[str]] = None,
        preprocessors: Iterable[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ):
        self.chain = Chain()
        self.branches = branches
        self.preprocessors = [*(preprocessors or ())]

        if friends:
            self.chain += friends

    def read(self, chunk: Chunk):
        chain = self.chain.copy()
        chain += chunk
        df = chain.concat(library="pd", reader_options={"branch_filter": self.branches})
        for preprocessor in self.preprocessors:
            if len(df) == 0:
                return None
            df = preprocessor(df)
        return df


class ToTensor:
    def __init__(self):
        self._columns: dict[
            str,
            tuple[
                npt.DTypeLike,
                list[tuple[str, Optional[int], Any, Optional[tuple[int, ...]]]],
            ],
        ] = {}
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
        pad_value: Any = 0,
        reshape: tuple[int, ...] = None,
    ):
        if self._current is None:
            raise RuntimeError("Call add() to specify a name before adding columns")
        self._columns[self._current][1].extend(
            (c, target, pad_value, reshape) for c in columns
        )
        return self

    def tensor(self, data: pd.DataFrame):
        dataset: dict[str, torch.Tensor] = {}
        for name, (dtype, columns) in self._columns.items():
            missing = [c for c, *_ in columns if c not in data]
            if missing:
                logging.warning(f"columns {missing} not found in dataframe")
                continue
            arrays = []
            for c, target, pad_value, reshape in columns:
                if data[c].dtype == "awkward":
                    if target is None:
                        target = int(np.max(data[c].ak.num()))
                    array = np.ma.filled(
                        data[c].ak.pad_none(target, clip=True).ak.to_numpy(), pad_value
                    ).astype(dtype)
                else:
                    array = data[c].to_numpy(dtype)
                    if len(array.shape) == 1 and len(columns) > 1:
                        array = array[:, np.newaxis]
                if reshape is not None:
                    array = array.reshape(reshape)
                arrays.append(array)
            if len(arrays) == 1:
                to_tensor = arrays[0]
            else:
                to_tensor = np.concatenate(arrays, axis=-1)
            if to_tensor.dtype == np.uint64:  # workaround for no "uint64" in torch
                to_tensor = to_tensor.view(np.uint32)
            dataset[name] = torch.from_numpy(to_tensor)
        return dataset
