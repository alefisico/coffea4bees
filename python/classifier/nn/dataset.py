from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.setting.torch import DataLoader as DLSetting

if TYPE_CHECKING:
    from torch.utils.data import Dataset


def entry_size(dataset: Dataset) -> int:
    from torch.utils.data import DataLoader

    entry = next(iter(DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)))
    if isinstance(entry, (list, tuple)):
        tensors = entry
    elif isinstance(entry, dict):
        tensors = entry.values()
    else:
        tensors = [entry]
    return sum(t.numel() * t.element_size() for t in tensors)


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
        kwargs["batch_size"] = int(DLSetting.batch_skim // (entry_size(dataset) / 4))
    return mp_loader(dataset, **kwargs)
