from __future__ import annotations

import argparse
import importlib
import os
import sys
from abc import ABC, abstractmethod
from collections import ChainMap, deque
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_CLASSIFIER = 'classifier'
_CONFIG = 'config'
_MAIN = 'main'


def _is_private(name: str):
    return name.startswith('_')


class ArgParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        kwargs['prog'] = kwargs.get('prog', None) or ''
        kwargs['add_help'] = False
        kwargs['conflict_handler'] = 'resolve'
        super().__init__(**kwargs)

    def remove_argument(self, *name_or_flags: str):
        self.add_argument(
            *name_or_flags, nargs=argparse.SUPPRESS, help=argparse.SUPPRESS, default=argparse.SUPPRESS)


class Task(ABC):
    argparser: ArgParser = NotImplemented
    defaults: dict[str] = NotImplemented

    @classmethod
    def __mod_name__(cls):
        return '.'.join(f'{cls.__module__}.{cls.__name__}'.split('.')[3:])

    def __init_subclass__(cls):
        defaults, parents, kwargs = [], [], {}
        for base in cls.__bases__:
            if issubclass(base, Task):
                if base.argparser is not NotImplemented:
                    parents.append(deepcopy(base.argparser))
                if base.defaults is not NotImplemented:
                    defaults.append(base.defaults)
        if 'argparser' in vars(cls):
            parents.append(cls.argparser)
            kwargs['prog'] = cls.argparser.prog or cls.__mod_name__()
            for k in ['usage', 'description', 'epilog']:
                kwargs[k] = getattr(cls.argparser, k, None)
        else:
            kwargs['prog'] = cls.__mod_name__()
        if 'defaults' in vars(cls):
            defaults.append(cls.defaults)
        cls.argparser = ArgParser(**kwargs, parents=parents)
        cls.defaults = (
            NotImplemented if len(defaults) == 0 else
            dict(ChainMap(*defaults[::-1])))

    def parse(self, args: list[str]):
        self.opts, _ = self.argparser.parse_known_args(args)
        if self.defaults is not NotImplemented:
            for k, v in self.defaults.items():
                if not hasattr(self.opts, k):
                    setattr(self.opts, k, v)
        return self

    @classmethod
    def help(cls):
        if cls.argparser is NotImplemented:
            raise ValueError(
                f'{cls.__name__}.argparser is not implemented')
        return cls.argparser.format_help()


class _Main(Task):
    @abstractmethod
    def run(self, parser: Parser):
        ...


class Parser:
    _tasks = list(map(lambda x: x.removesuffix('.py'),
                  filter(lambda x: x.endswith('.py') and not _is_private(x),
                         os.listdir(Path(__file__).parent/f'../{_CONFIG}/{_MAIN}'))))
    _keys = ['dataset', 'model']
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
        except:
            ...
        if _mod is not None and clsname != '*':
            try:
                _cls = getattr(_mod, clsname)
            except:
                ...
        return _mod, _cls

    def _fetch_all(self):
        self.mods: dict[str, list[Task]] = {}
        for cat in self._keys:
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
                    if not issubclass(cls, Task):
                        raise TypeError(
                            f'Class "{clsname}" is not a subclass of Task')
                    else:
                        self.mods[cat].append(cls().parse(opts))

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
        self.main: _Main = cls()

    def run(self):
        self.main.parse(self.args[_MAIN][1])
        if self.args[_MAIN][0] != 'help':
            self._fetch_all()
        self.main.run(self)
