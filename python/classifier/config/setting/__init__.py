from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Mapping

import fsspec
from classifier.task.state import Cascade, _share_global_state
from classifier.typetools import dict_proxy

if TYPE_CHECKING:
    from base_class.system.eos import EOS

_SETTING = "setting"
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


class setup(Cascade):
    @classmethod
    def parse(cls, opts: list[str]):
        from classifier.task import EntryPoint, parse

        for opt in opts:
            configs = parse.mapping(f"file:///{opt}")
            for k, v in configs.items():
                _, k = EntryPoint._fetch_module(k, _SETTING)
                if isinstance(v, Mapping):
                    dict_proxy(k).update(v)
                elif isinstance(v, list):
                    k.parse(v)

    @classmethod
    def help(cls):
        infos = [
            _SPECIAL,
            f"usage: {cls.__mod_name__()} FILE [FILE ...]",
            "",
            "Setup all settings from input files.",
            "",
        ]
        return "\n".join(infos)


class IO(Cascade):
    timestamp: str = "%Y-%m-%dT%H-%M-%S"

    output: EOS = "./{main}-{timestamp}/"

    file_logs: str = "logs.html"
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

    # performance
    max_resend: int = 1
    reconnect_delay: float = 0.1

    # builtins
    logging_level: int = 20
    socket_timeout: float = None

    @classmethod
    def set__socket_timeout(cls, value: float):
        import socket

        socket.setdefaulttimeout(value)
