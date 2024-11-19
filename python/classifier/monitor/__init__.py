import importlib
import logging as _logging

from classifier.config import setting as cfg

from ..config.state import MonitorInfo
from ..process import status
from .backends import Platform
from .core import Monitor, Recorder, connect_to_monitor, wait_for_monitor
from .template import Index

__all__ = [
    "Platform",
    "Index",
    "Monitor",
    "Recorder",
    "wait_for_monitor",
    "setup_monitor",
    "disable_monitor",
    "setup_reporter",
]

_PKG = "classifier.monitor"
_BACKENDS = "backends"


def setup_monitor():
    Monitor.start()

    for backend in MonitorInfo.backends:
        mod = importlib.import_module(f"{_PKG}.{_BACKENDS}.{backend}")
        if hasattr(mod, "setup_backend"):
            mod.setup_backend()

    for component in MonitorInfo.components:
        mod = importlib.import_module(f"{_PKG}.{component}")
        if hasattr(mod, "setup_monitor"):
            mod.setup_monitor()

    _logging.info(f"Monitor is running at {cfg.Monitor.raw__address}")


def setup_reporter():
    connect_to_monitor()

    for component in MonitorInfo.components:
        mod = importlib.import_module(f"{_PKG}.{component}")
        if hasattr(mod, "setup_reporter"):
            mod.setup_reporter()

    if cfg.Monitor.log_show_connection:
        _logging.info(f"Connected to Monitor {cfg.Monitor.raw__address}")


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
