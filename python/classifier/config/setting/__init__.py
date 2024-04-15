from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING

import fsspec
from classifier.task.state import Cascade, _share_global_state

if TYPE_CHECKING:
    from base_class.system.eos import EOS


class save(Cascade):
    @classmethod
    def parse(cls, opts: list[str]):
        with fsspec.open(opts[0], "wb") as f:
            pickle.dump(_share_global_state(), f)

    @classmethod
    def help(cls):
        infos = [
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
            f"usage: {cls.__mod_name__()} INPUT [INPUT ...]",
            "",
            "Load global states from files.",
            "",
        ]
        return "\n".join(infos)


class IO(Cascade):
    timestamp: str = "%Y-%m-%dT%H-%M-%S"

    output: EOS = "./{main}-{timestamp}/"
    monitor: EOS = "diagnostics"
    profiler: EOS = "profiling"

    file_states: str = "states.pkl"
    file_metadata: str = "metadata.json"

    @classmethod
    def _generate_path(cls, value: str):
        return (cls.output / value).mkdir(recursive=True)

    @classmethod
    def set__output(cls, value: str):
        from ..state import RunInfo

        if value is None:
            return os.devnull
        return value.format(
            main=RunInfo.main_task,
            timestamp=RunInfo.startup_time.strftime(cls.timestamp),
        )

    @classmethod
    def get__output(cls, value: str):
        from base_class.system.eos import EOS

        return EOS(value).mkdir(recursive=True)

    @classmethod
    def get__monitor(cls, value: str):
        return cls._generate_path(value)

    @classmethod
    def get__profiler(cls, value: str):
        return cls._generate_path(value)


class Monitor(Cascade):
    enable: bool = True
    address: str = None
    port: int = 12345

    # backends
    console_enable: bool = True
    console_update_interval: float = 1.0  # seconds
    console_fps: int = 10

    web_enable: bool = False

    # components
    log_enable: bool = True

    progress_enable: bool = True

    usage_enable: bool = True
    usage_update_interval: float = 1.0  # seconds
    usage_gpu: bool = True
    usage_gpu_force_torch: bool = False

    # records
    file_meta: str = "meta.json"
    file_log: str = "logs.html"
    file_usage: str = "usage.json"

    # performance
    max_resend: int = 1
    reconnect_delay: float = 0.1  # seconds

    # builtins
    logging_level: int = 20
    socket_timeout: float = None
    warnings_ignore: bool = True

    @classmethod
    def set__socket_timeout(cls, value: float):
        import socket

        socket.setdefaulttimeout(value)

    @classmethod
    def set__warnings_ignore(cls, value: bool):
        import warnings

        if value:
            warnings.filterwarnings("ignore")
        else:
            warnings.filterwarnings("default")
