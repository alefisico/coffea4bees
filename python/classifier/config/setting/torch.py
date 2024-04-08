from __future__ import annotations

from classifier.task import Cascade


class DataLoader(Cascade):
    batch_skim: int = 2**17
    batch_eval: int = 2**15
    num_workers: int = 0
