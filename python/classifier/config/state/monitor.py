from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.task import GlobalState

if TYPE_CHECKING:
    from classifier.monitor import TrainingBenchmark


class Classifier(GlobalState):
    benchmark: list[TrainingBenchmark] = []
