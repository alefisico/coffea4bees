from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from ..config.setting.torch import DataLoader as cfg
from ..monitor.progress import MessageType, Progress


class DataLoaderWithProgress(DataLoader):
    _progress_msg: MessageType

    def _progress_iter(self):
        progress = Progress.new(len(self), self._progress_msg)
        for i, data in enumerate(super().__iter__()):
            yield data
            progress.update(i + 1)

    def __iter__(self):
        if hasattr(self, "_progress_msg") and self._progress_msg is not None:
            return self._progress_iter()
        else:
            return super().__iter__()


def simple_loader(
    dataset: Dataset, report_progress: MessageType = None, **kwargs
) -> DataLoader:
    kwargs.setdefault("num_workers", cfg.num_workers)
    kwargs.setdefault("persistent_workers", cfg.persistent_workers)
    loader = DataLoaderWithProgress(dataset, **kwargs)
    loader._progress_msg = report_progress
    if loader.num_workers != 0:
        from ..process import status

        loader.multiprocessing_context = status.context
    return loader


def skim_loader(dataset: Dataset, report_progress: MessageType = None, **kwargs):
    if "batch_size" not in kwargs:
        kwargs["batch_size"] = cfg.batch_skim
    return simple_loader(dataset, report_progress=report_progress, **kwargs)
