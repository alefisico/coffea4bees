from __future__ import annotations

import json
import logging
import os
import sys
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import fsspec

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
_FROM = "from"

_MODULE = "module"
_OPTION = "option"

_DASH = "-"


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
    _preserved = [*(f"--{k}" for k in _keys), f"--{_FROM}"]

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

    @classmethod
    def _expand_module(cls, data: dict):
        import shlex

        from . import parse

        mod = data[_MODULE]
        opts = []
        for opt in data.get(_OPTION, []):
            if opt is None:
                opts.append("")
            elif isinstance(opt, str):
                opts.extend(shlex.split(opt))
            else:
                opts.append(parse.escape(opt))
        return mod, opts

    def _expand(self, *files: str, fetch_main: bool = False):
        from . import parse

        for file in files:
            args = parse.mapping(file, "file")
            if fetch_main and _MAIN in args:
                self.args[_MAIN] = self._expand_module(args[_MAIN])
            for cat in self._keys:
                if cat in args:
                    for arg in args[cat]:
                        self.args[cat].append(self._expand_module(arg))

    def __init__(self, argv: list[str] = None, initializer: Callable[[], None] = None):
        from ..config.state import RunInfo

        if initializer is not None:
            initializer()
        if argv is None:
            argv = sys.argv.copy()

        self.entrypoint = Path(argv[0]).name
        self.cmd = " ".join(argv)
        self.args: dict[str, list[tuple[str, list[str]]]] = {k: [] for k in self._keys}

        args = deque(argv[1:])
        if len(args) == 0:
            raise ValueError(f"No task specified")
        arg = args.popleft()
        self.args[_MAIN] = arg, self._fetch_subargs(args)
        if arg == _FROM:
            self._expand(*self.args[_MAIN][1], fetch_main=True)
        while len(args) > 0:
            cat = args.popleft().removeprefix("--")
            mod = args.popleft()
            opts = self._fetch_subargs(args)
            if cat == _FROM:
                self._expand(mod)
            else:
                self.args[cat].append((mod, opts))

        main: str = self.args[_MAIN][0]
        if main not in self._tasks:
            raise ValueError(
                f'The first argument must be one of {self._tasks}, got "{main}"'
            )
        RunInfo.main_task = main

        cls: type[Main] = self._fetch_module(f"{self.args[_MAIN][0]}.Main", _MAIN)[1]
        if cls is None:
            raise AttributeError(f'Task "{self.args[_MAIN][0]}" not found')

        if not cls._no_load:
            self._fetch_all()

        self.main: Main = new(cls, self.args[_MAIN][1])

        from ..config.setting import Monitor as cfg

        if not cls._no_monitor and cfg.enable:
            if cfg.address is None:
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
                logging.info(f"Connecting to Monitor {cfg.address}:{cfg.port}")

    def run(self, reproducible: Callable):
        from ..config.setting import IO, save
        from ..process.monitor import Recorder, wait_for_monitor

        meta = self.main.run(self)
        wait_for_monitor()

        if self.main.flag("save_state"):
            save.parse([IO.output / IO.file_states])
        Recorder.dump()

        if meta is not None:
            from base_class.utils.json import DefaultEncoder

            meta |= {
                "command": self.cmd,
                "reproducible": reproducible(),
            }
            with fsspec.open(
                IO.output / IO.file_metadata,
                "wt",
            ) as f:
                json.dump(meta, f, cls=DefaultEncoder)


# main


class Main(Task):
    _no_monitor = False
    _no_load = False

    argparser = ArgParser()
    argparser.add_argument(
        "--save-state",
        action="store_true",
        help="save global states to the output directory",
    )

    @interface
    def run(self, parser: EntryPoint) -> Optional[dict[str]]: ...
