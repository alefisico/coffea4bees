import importlib

from ..config.state import MonitorInfo
from ..process import status
from .backends import Platform

__all__ = [
    "Platform",
]
_PKG = "classifier.monitor"


def setup_monitor():
    for backend in MonitorInfo.backends:
        mod = importlib.import_module(f"{_PKG}.backends.{backend}")
        if hasattr(mod, "setup_backend"):
            mod.setup_backend()

    for component in MonitorInfo.components:
        mod = importlib.import_module(f"{_PKG}.{component}")
        if hasattr(mod, "setup_monitor"):
            mod.setup_monitor()


def setup_reporter():
    for component in MonitorInfo.components:
        mod = importlib.import_module(f"{_PKG}.{component}")
        if hasattr(mod, "setup_reporter"):
            mod.setup_reporter()


status.initializer.add_unique(setup_reporter)
