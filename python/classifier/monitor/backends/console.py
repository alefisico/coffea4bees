import time
from threading import Lock, Thread
from typing import Callable

from rich.console import Console
from rich.live import Live
from rich.table import Table

from ...config.setting import Monitor as cfg


class Dashboard:
    console: Console = None
    layout: Table = None

    _lock: Lock = None
    _callbacks: list[Callable[[], None]] = []

    @classmethod
    def start(cls):
        with Live(
            cls.layout,
            refresh_per_second=cfg.console_fps,
            transient=True,
            console=cls.console,
        ):
            while True:
                with cls._lock:
                    for callback in cls._callbacks:
                        callback()
                time.sleep(cfg.console_update_interval)

    @classmethod
    def add(cls, callback: Callable[[], None]):
        with cls._lock:
            cls._callbacks.append(callback)


def setup_backend():
    Dashboard.console = Console(markup=True)
    Dashboard.layout = Table.grid()
    Dashboard._lock = Lock()
    Thread(target=Dashboard.start, daemon=True).start()
