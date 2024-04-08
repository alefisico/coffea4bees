from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import fsspec
from classifier.task.state import Cascade, _share_global_state

if TYPE_CHECKING:
    from base_class.system.eos import EOS

_SPECIAL = "[red]\[special][/red]"


class save(Cascade):
    @classmethod
    def parse(cls, opts: list[str]):
        with fsspec.open(opts[0], "wb") as f:
            pickle.dump(_share_global_state(), f)

    @classmethod
    def help(cls):
        infos = [
            _SPECIAL,
            f"usage: {cls.__mod_name__()} OUTPUT",
            "",
            "Save global states to file.",
            "",
        ]
        return "\n".join(infos)


class load(Cascade):
    @classmethod
    def parse(cls, opts: list[str]):
        for opt in opts:
            with fsspec.open(opt, "rb") as f:
                pickle.load(f)()

    @classmethod
    def help(cls):
        infos = [
            _SPECIAL,
            f"usage: {cls.__mod_name__()} INPUT [INPUT ...]",
            "",
            "Load global states from files.",
            "",
        ]
        return "\n".join(infos)


class IO(Cascade):
    timestamp: str = "%Y-%m-%dT%H-%M-%S"

    output: EOS = "./{main}-{timestamp}/"

    file_states: str = "states.pkl"
    file_metadata: str = "metadata.json"

    @classmethod
    def set__output(cls, value: str):
        from ..state import RunInfo

        return value.format(
            main=RunInfo.main_task,
            timestamp=RunInfo.startup_time.strftime(cls.timestamp),
        )

    @classmethod
    def get__output(cls, value: str):
        from base_class.system.eos import EOS

        return EOS(value).mkdir(recursive=True)


class Monitor(Cascade):
    address: str = None
    port: int = 12345

    # backends
    use_console: bool = True
    use_web: bool = False

    # components
    show_log: bool = True
    show_progress: bool = True

    track_usage: bool = True
    track_usage_interval: float = 1.0  # seconds

    # records
    dir_records: str = "diagnostic"
    file_meta: str = "meta.json"
    file_logs: str = "logs.html"
    file_usage: str = "usage.json"

    # performance
    max_resend: int = 1
    reconnect_delay: float = 0.1  # seconds

    # builtins
    logging_level: int = 20
    socket_timeout: float = None

    @classmethod
    def set__socket_timeout(cls, value: float):
        import socket

        socket.setdefaulttimeout(value)
