from __future__ import annotations

from typing import Protocol

from .special import interface
from .task import Task


class Analysis(Task):
    @interface
    def analyze(self, output: dict = None) -> list[Analyzer]:
        """
        Prepare analyzers.
        """
        ...


class Analyzer(Protocol):
    def __call__(self):
        """
        Run analysis.
        """
        ...
