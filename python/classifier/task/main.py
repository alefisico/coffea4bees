from __future__ import annotations

import json
import logging
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import fsspec
import rich.terminal_theme as themes

from ..utils import import_
from .dataset import Dataset
from .model import Model
from .special import interface, new
from .state import Cascade, _is_private
from .task import ArgParser, Task

if TYPE_CHECKING:
    from types import ModuleType


_CLASSIFIER = "classifier"
_CONFIG = "config"
_MAIN = "main"


class EntryPoint:
    _tasks = list(
        map(
            lambda x: x.removesuffix(".py"),
            filter(
                lambda x: x.endswith(".py") and not _is_private(x),
                os.listdir(Path(__file__).parent / f"../{_CONFIG}/{_MAIN}"),
            ),
        )
    )
    _keys = {
        "setting": Cascade,
        "dataset": Dataset,
        "model": Model,
    }
    _preserved = [f"--{k}" for k in _keys]

    @classmethod
    def _fetch_subargs(cls, args: deque):
        subargs = []
        while len(args) > 0 and args[0] not in cls._preserved:
            subargs.append(args.popleft())
        return subargs

    @classmethod
    def _fetch_module_name(cls, module: str, key: str):
        mods = [_CLASSIFIER, _CONFIG, key, *module.split(".")]
        return ".".join(mods[:-1]), mods[-1]

    @classmethod
    def _fetch_module(cls, module: str, key: str) -> tuple[ModuleType, type[Task]]:
        return import_(*cls._fetch_module_name(module, key))

    def _fetch_all(self):
        self.mods: dict[str, list[Task]] = {}
        for cat in self._keys:
            target = self._keys[cat]
            self.mods[cat] = []
            for imp, opts in self.args[cat]:
                modname, clsname = self._fetch_module_name(imp, cat)
                mod, cls = self._fetch_module(imp, cat)
                if mod is None:
                    raise ModuleNotFoundError(f'Module "{modname}" not found')
                elif cls is None:
                    raise AttributeError(
                        f'Class "{clsname}" not found in module "{modname}"'
                    )
                else:
                    if not issubclass(cls, target):
                        raise TypeError(f'Class "{clsname}" is not a subclass of Task')
                    else:
                        self.mods[cat].append(new(cls, opts))

    def __init__(self):
        from ..config.state import RunInfo

        RunInfo.startup_time = datetime.now()
        RunInfo.main_task = sys.argv[1]

        self.entrypoint = Path(sys.argv[0]).name
        self.cmd = " ".join(sys.argv)
        self.args: dict[str, list[tuple[str, list[str]]]] = {k: [] for k in self._keys}

        args = deque(sys.argv[1:])
        if len(args) == 0:
            raise ValueError(f"No task specified")
        arg = args.popleft()
        if arg not in self._tasks:
            raise ValueError(
                f'The first argument must be one of {self._tasks}, got "{arg}"'
            )
        else:
            self.args[_MAIN] = arg, self._fetch_subargs(args)
        while len(args) > 0:
            cat = args.popleft().removeprefix("--")
            mod = args.popleft()
            opts = self._fetch_subargs(args)
            self.args[cat].append((mod, opts))

        _, cls = self._fetch_module(f"{self.args[_MAIN][0]}.Main", _MAIN)
        if cls is None:
            raise AttributeError(f'Task "{self.args[_MAIN][0]}" not found')

        self._fetch_all()
        self.main: Main = new(cls, self.args[_MAIN][1])
        if not self.main._no_monitor:
            from ..config.setting import Monitor as Setting

            if Setting.address is None:
                from ..monitor import setup_monitor
                from ..process.monitor import Monitor

                Monitor().start()
                setup_monitor()
                address, port = Monitor.current()._address
                logging.info(f"Started Monitor at {address}:{port}")
            else:
                from ..monitor import setup_reporter
                from ..process.monitor import connect_to_monitor

                connect_to_monitor()
                setup_reporter()
                logging.info(f"Connecting to Monitor {Setting.address}:{Setting.port}")

    def run(self, reproducible: Callable):
        from ..config.setting import IO as IOSetting
        from ..process.monitor import wait_for_reporter

        meta = self.main.run(self)
        wait_for_reporter()

        if self.main.flag("save_state"):
            from ..config.setting import save

            save.parse([IOSetting.output / IOSetting.file_states])

        if self.main.flag("save_logs"):
            from ..monitor.logging import Log

            if Log.console is not None:
                with fsspec.open(
                    IOSetting.output / IOSetting.file_logs,
                    "wt",
                ) as f:
                    f.write(Log.console.export_html(theme=themes.MONOKAI))

        if meta is not None:
            from base_class.utils.json import DefaultEncoder

            meta |= {
                "command": self.cmd,
                "reproducible": reproducible(),
            }
            with fsspec.open(
                IOSetting.output / IOSetting.file_metadata,
                "wt",
            ) as f:
                json.dump(meta, f, cls=DefaultEncoder)


# main


class Main(Task):
    _no_monitor = False

    argparser = ArgParser()
    argparser.add_argument(
        "--save-state",
        action="store_true",
        help="save global states to the output directory",
    )
    argparser.add_argument(
        "--save-logs",
        action="store_true",
        help="save logs to the output directory",
    )

    @interface
    def run(self, parser: EntryPoint) -> Optional[dict[str]]: ...
