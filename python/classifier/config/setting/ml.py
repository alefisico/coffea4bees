from __future__ import annotations

from classifier.task import GlobalSetting


class DataLoader(GlobalSetting):
    "Default torch.DataLoader arguments"

    batch_skim: int = 2**17
    "batch size for skimming"
    batch_eval: int = 2**15
    "batch size for evaluation"
    pin_memory: bool = True
    num_workers: int = 0
    persistent_workers: bool = True

    @classmethod
    def get__persistent_workers(cls, value: bool) -> bool:
        if cls.num_workers == 0:
            return False
        return value


class KFold(GlobalSetting):
    "KFolding dataset splitter"

    offset: str = "offset"
    "key of the offset in the input batch"
    offset_dtype: str = "uint64"
    "dtype of the offset tensor"


class Training(GlobalSetting):
    "Multistage training"

    disable_benchmark: bool = False
    "disable unrequired benchmark steps"
