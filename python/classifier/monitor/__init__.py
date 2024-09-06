import importlib

from ..config.state import MonitorInfo
from ..process import status
from .backends import Platform
from .template import Index

__all__ = [
    "Platform",
    "Index",
]
_PKG = "classifier.monitor"
_BACKENDS = "backends"


def setup_monitor():
    for backend in MonitorInfo.backends:
        mod = importlib.import_module(f"{_PKG}.{_BACKENDS}.{backend}")
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


def disable_monitor():
    for backend in MonitorInfo.backends:
        mod = importlib.import_module(f"{_PKG}.{_BACKENDS}.{backend}")
        if hasattr(mod, "disable_backend"):
            mod.disable_backend()

    for component in MonitorInfo.components:
        mod = importlib.import_module(f"{_PKG}.{component}")
        if hasattr(mod, "disable_monitor"):
            mod.disable_monitor()


status.initializer.add_unique(setup_reporter)
