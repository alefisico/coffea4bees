from __future__ import annotations

from classifier.task import Cascade


class DataLoader(Cascade):
    batch_io: int = 1_000_000
    batch_eval: int = 2**15
    num_workers: int = 0
