
from __future__ import annotations


import torch
from torch.utils.data import DataLoader, Dataset


class MultiTensorDataset(Dataset):
    ...  # TODO


class NamedDataLoader(DataLoader):
    def __init__(
            self,
            columns: list[str],
            # TODO custom dataset using multiple chunks of tensors without copying
            dataset: Dataset,
            **kwargs):
        """
        kwargs are passed to :class:`torch.utils.data.DataLoader`
        """
        super().__init__(dataset=dataset, **kwargs)
        self.columns = columns

    def __iter__(self):
        for batch in super().__iter__():
            batch: tuple[torch.Tensor, ...]
            yield dict(zip(self.columns, batch))
