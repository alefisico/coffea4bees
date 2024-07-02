from __future__ import annotations

import json
import logging
import os
import sys
from collections import deque
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import fsspec

from ..utils import import_
from .analysis import Analysis
from .dataset import Dataset
from .model import Model
from .special import interface, new
from .state import Cascade, _is_private
from .task import _DASH, Task

if TYPE_CHECKING:
    from types import ModuleType


_CLASSIFIER = "classifier"
_CONFIG = "config"
_MAIN = "main"
_FROM = "from"
_TEMPLATE = "template"

_MODULE = "module"
_OPTION = "option"


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
        "analysis": Analysis,
    }
    _preserved = [
        *(f"{_DASH}{k}" for k in chain(_keys, (_FROM, _TEMPLATE))),
    ]

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
                        raise TypeError(
                            f'Class "{clsname}" is not a subclass of "{target.__name__}"'
                        )
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

    def _expand(self, *files: str, fetch_main: bool = False, formatter: str = None):
        from . import parse

        for file in files:
            args = parse.mapping(file, "file", formatter)
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
            raise ValueError("No task specified")
        arg = args.popleft()
        self.args[_MAIN] = arg, self._fetch_subargs(args)
        if arg == _FROM:
            self._expand(*self.args[_MAIN][1], fetch_main=True)
        elif arg == _TEMPLATE:
            self._expand(
                *self.args[_MAIN][1][1:],
                fetch_main=True,
                formatter=self.args[_MAIN][1][0],
            )
        while len(args) > 0:
            cat = args.popleft().removeprefix(_DASH)
            mod = args.popleft()
            opts = self._fetch_subargs(args)
            if cat == _FROM:
                self._expand(mod, *opts)
            elif cat == _TEMPLATE:
                self._expand(*opts, formatter=mod)
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

        if cls.prelude is not NotImplemented:
            cls.prelude()

        if not cls._no_init:
            self._fetch_all()

        self.main: Main = new(cls, self.args[_MAIN][1])

        from ..config import setting as cfg

        if cfg.Monitor.enable:
            host, port = cfg.Monitor.address
            if host is None:
                from ..monitor import setup_monitor
                from ..process.monitor import Monitor

                Monitor().start()
                setup_monitor()
                address = Monitor.current()._address
                if isinstance(address, tuple):
                    address = f"{address[0]}:{address[1]}"
                logging.info(f"Started Monitor at {address}")
            else:
                from ..monitor import setup_reporter
                from ..process.monitor import connect_to_monitor

                connect_to_monitor()
                setup_reporter()
                address = host if port is None else f"{host}:{port}"
                logging.info(f"Connecting to Monitor {address}")

    def run(self, reproducible: Callable = None):
        from ..config import setting as cfg
        from ..config.main.analyze import run_analyzer
        from ..process.monitor import Recorder, wait_for_monitor

        # run main task
        result = self.main.run(self)
        # run analysis on result
        if cfg.Analysis.enable:
            analysis = run_analyzer(self, result)
            if analysis:
                result = result or {}
                result["analysis"] = analysis
        # wait for monitor
        wait_for_monitor()
        # dump state
        if (not self.main._no_state) and (not cfg.IO.states.is_null):
            cfg.save.parse([cfg.IO.states])
        # dump diagnostics
        Recorder.dump()
        # dump result
        if (result is not None) and (not cfg.IO.result.is_null):
            from base_class.utils.json import DefaultEncoder

            result["command"] = self.cmd
            if reproducible is not None:
                result["reproducible"] = reproducible()
            with fsspec.open(cfg.IO.result, "wt") as f:
                json.dump(result, f, cls=DefaultEncoder)


# main


class Main(Task):
    _no_state = False
    _no_init = False

    @classmethod
    @interface(optional=True)
    def prelude(cls): ...

    @interface
    def run(self, parser: EntryPoint) -> Optional[dict[str]]: ...
