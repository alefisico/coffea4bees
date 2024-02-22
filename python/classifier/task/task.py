from __future__ import annotations

import argparse
from collections import ChainMap
from copy import deepcopy
from typing import TypeVar

_InterfaceT = TypeVar('_InterfaceT')


class _Interface:
    def __init__(self, func):
        self.func = func

    def __get__(self, _, owner):
        import inspect
        signature = (
            str(inspect.signature(self.func))
            .replace("'", "")
            .replace('"', ''))
        raise NotImplementedError(
            f'Task interface {owner.__name__}.{self.func.__name__}{signature} is not implemented')


def interface(func: _InterfaceT) -> _InterfaceT:
    return _Interface(func)


class ArgParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        kwargs['prog'] = kwargs.get('prog', None) or ''
        kwargs['add_help'] = False
        kwargs['conflict_handler'] = 'resolve'
        kwargs['formatter_class'] = argparse.ArgumentDefaultsHelpFormatter
        super().__init__(**kwargs)

    def remove_argument(self, *name_or_flags: str):
        self.add_argument(
            *name_or_flags, nargs=argparse.SUPPRESS, help=argparse.SUPPRESS, default=argparse.SUPPRESS)


class Task:
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
