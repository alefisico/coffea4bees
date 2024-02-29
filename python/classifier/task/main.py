from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import fsspec

from ..config.setting.default import IO as IOSetting
from ..process.state import Cascade, _is_private
from .dataset import Dataset
from .model import Model
from .task import ArgParser, Task, interface

if TYPE_CHECKING:
    from types import ModuleType


_CLASSIFIER = 'classifier'
_CONFIG = 'config'
_MAIN = 'main'


def _new(cls: type[Task], opts: list[str]):
    obj = object.__new__(cls)
    obj.parse(opts)
    obj.__init__()
    return obj


class EntryPoint:
    _tasks = list(map(lambda x: x.removesuffix('.py'),
                  filter(lambda x: x.endswith('.py') and not _is_private(x),
                         os.listdir(Path(__file__).parent/f'../{_CONFIG}/{_MAIN}'))))
    _keys = {
        'setting': Cascade,
        'dataset': Dataset,
        'model': Model,
    }
    _preserved = [f'--{k}' for k in _keys]

    @classmethod
    def _fetch_subargs(cls, args: deque):
        subargs = []
        while len(args) > 0 and args[0] not in cls._preserved:
            subargs.append(args.popleft())
        return subargs

    @classmethod
    def _fetch_module_name(cls, module: str, key: str):
        mods = [_CLASSIFIER, _CONFIG, key, *module.split('.')]
        return '.'.join(mods[:-1]), mods[-1]

    @classmethod
    def _fetch_module(cls, module: str, key: str) -> tuple[ModuleType, type[Task]]:
        modname, clsname = cls._fetch_module_name(module, key)
        _mod, _cls = None, None
        try:
            _mod = importlib.import_module(modname)
        except ModuleNotFoundError:
            ...
        except Exception as e:
            logging.error(e)
        if _mod is not None and clsname != '*':
            try:
                _cls = getattr(_mod, clsname)
            except AttributeError:
                ...
            except Exception as e:
                logging.error(e)
        return _mod, _cls

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
                        f'Class "{clsname}" not found in module "{modname}"')
                else:
                    if not issubclass(cls, target):
                        raise TypeError(
                            f'Class "{clsname}" is not a subclass of Task')
                    else:
                        self.mods[cat].append(_new(cls, opts))

    def __init__(self):
        self.entrypoint = Path(sys.argv[0]).name
        self.cmd = ' '.join(sys.argv)
        self.args: dict[str, list[tuple[str, list[str]]]] = {
            k: [] for k in self._keys}

        args = deque(sys.argv[1:])
        if len(args) == 0:
            raise ValueError(f'No task specified')
        arg = args.popleft()
        if arg not in self._tasks:
            raise ValueError(
                f'The first argument must be one of {self._tasks}, got "{arg}"')
        else:
            self.args[_MAIN] = arg, self._fetch_subargs(args)
        while len(args) > 0:
            cat = args.popleft().removeprefix('--')
            mod = args.popleft()
            opts = self._fetch_subargs(args)
            self.args[cat].append((mod, opts))

        _, cls = self._fetch_module(f'{self.args[_MAIN][0]}.Main', _MAIN)
        if cls is None:
            raise AttributeError(f'Task "{self.args[_MAIN][0]}" not found')
        self.main: Main = cls()

    def run(self, reproducible: Callable):
        self.main.parse(self.args[_MAIN][1])
        self._fetch_all()
        meta = self.main.run(self)

        if hasattr(self.main.opts, 'save_state') and self.main.opts.save_state:
            from ..config.setting.cache import save
            save.parse([IOSetting.output/'state.pkl'])

        if meta is not None:
            from base_class.utils.json import DefaultEncoder

            meta |= {
                'command': self.cmd,
                'reproducible': reproducible(),
            }
            with fsspec.open(IOSetting.output/f'{self.args[_MAIN][0]}.json', 'wt') as f:
                json.dump(meta, f, cls=DefaultEncoder)


# main

class Main(Task):
    argparser = ArgParser()
    argparser.add_argument(
        '--save-state', action='store_true', help='save global states to the output directory')

    @interface
    def run(self, parser: EntryPoint) -> Optional[dict[str]]:
        ...
