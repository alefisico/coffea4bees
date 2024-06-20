from __future__ import annotations

import getpass
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
    report: EOS = "report"
    profiler: EOS = "profiling"

    states: EOS = "states.pkl"
    result: EOS = "result.json"

    @classmethod
    def _generate_path(cls, value: str):
        return (cls.output / value).mkdir(recursive=True)

    @classmethod
    def set__output(cls, value: str):
        from ..state import RunInfo

        if value is None:
            return os.devnull
        return value.format(
            user=getpass.getuser(),
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
    def get__report(cls, value: str):
        return cls._generate_path(value)

    @classmethod
    def get__profiler(cls, value: str):
        return cls._generate_path(value)

    @classmethod
    def get__states(cls, value: str):
        return cls.output / value

    @classmethod
    def get__result(cls, value: str):
        return cls.output / value


class Monitor(Cascade):
    enable: bool = True
    file: str = "meta.json"
    address: tuple[str, int] = ":10200"

    # performance
    retry_max: int = 1
    reconnect_delay: float = 0.1  # seconds

    # builtins
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

    @classmethod
    def get__address(cls, value: int | str):
        if isinstance(value, int):
            return None, value
        if value is None:
            return None, None
        parts = value.rsplit(":", 1)
        if len(parts) == 2:
            try:
                port = int(parts[1])
                host = parts[0] or None
                return host, port
            except:
                pass
        return value or None, None


class Analysis(Cascade):
    enable: bool = True
    max_workers: int = 1
