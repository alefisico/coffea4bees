from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor


class TrainingBenchmark(Protocol):
    def __call__(self, pred: dict[str, Tensor], **metadata) -> dict[str]:
        ...
