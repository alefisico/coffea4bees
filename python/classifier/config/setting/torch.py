from __future__ import annotations

from classifier.task import Cascade


class DataLoader(Cascade):
    batch_skim: int = 2**17
    batch_eval: int = 2**15
    num_workers: int = 0


class KFold(Cascade):
    offset: str = "offset"
    offset_dtype: str = "uint64"


class Training(Cascade):
    disable_benchmark: bool = False
