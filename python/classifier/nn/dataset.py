from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.setting.torch import DataLoader as DLSetting

if TYPE_CHECKING:
    from torch.utils.data import Dataset


def mp_loader(dataset: Dataset, **kwargs):
    from torch.utils.data import DataLoader

    kwargs.setdefault("num_workers", DLSetting.num_workers)
    loader = DataLoader(dataset, **kwargs)
    if loader.num_workers != 0:
        from ..process import status

        loader.multiprocessing_context = status.context
    return loader


def skim_loader(dataset: Dataset, **kwargs):
    if "batch_size" not in kwargs:
        kwargs["batch_size"] = DLSetting.batch_skim
    return mp_loader(dataset, **kwargs)
