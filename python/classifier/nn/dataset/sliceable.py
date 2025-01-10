from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator, Generic, TypeVar

import numpy as np
import torch
from base_class.math.random import SeedLike, _seed
from torch.utils.data import Dataset

_BatchT = TypeVar("_BatchT")


class SliceableDataset(Dataset, ABC, Generic[_BatchT]):
    @abstractmethod
    def __getitem__(self, indices: torch.Tensor | slice) -> _BatchT: ...

    @abstractmethod
    def __len__(self) -> int: ...


class NamedTensorDataset(SliceableDataset[dict[str, torch.Tensor]]):
    def __init__(self, **tensors: torch.Tensor):
        self._tensors = tensors

    def __len__(self):
        return self._tensors["x"].shape[0]

    def __getitem__(self, indices: torch.Tensor | slice):
        batch = {k: v[indices] for k, v in self._tensors.items()}
        if isinstance(indices, slice):
            for k in batch:
                batch[k] = batch[k].clone()
        return batch

    def concat(self, *datasets: NamedTensorDataset):
        return NamedTensorDataset(
            **{k: torch.cat([t._tensors[k] for t in datasets]) for k in self._tensors}
        )

    def subset(self, indices: torch.Tensor):
        return NamedTensorDataset(**self[indices])


class SliceLoaderLite(Generic[_BatchT]):
    def __init__(
        self,
        dataset: SliceableDataset[_BatchT],
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: SeedLike = None,
        **kwargs,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed if seed is None else _seed(seed)
        self.kwargs = kwargs

    def __iter__(self) -> Generator[_BatchT, None, None]:
        data = self.dataset
        ndata = len(data)
        nbatch = self.batch_size
        indices = None
        if self.shuffle:
            rng = np.random.Generator(np.random.PCG64(self.seed))
            indices = torch.from_numpy(rng.permutation(ndata))
        for i in range(0, ndata, nbatch):
            if self.drop_last and (i + nbatch) > ndata:
                break
            if indices is not None:
                yield data[indices[i : i + nbatch]]
            else:
                yield data[i : i + nbatch]
