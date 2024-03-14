from __future__ import annotations

from logging import Logger
from typing import TypeVar

__all__ = ["interface", "new", "TaskBase", "Static", "Unique"]

_InterfaceT = TypeVar("_InterfaceT")


class _Interface:
    def __init__(self, func):
        self.func = func

    def __get__(self, _, owner):
        import inspect

        signature = str(inspect.signature(self.func)).replace("'", "").replace('"', "")
        raise NotImplementedError(f"{owner.__name__}.{self.func.__name__}{signature}")


def interface(func: _InterfaceT) -> _InterfaceT:
    return _Interface(func)


class TaskBase:
    @interface
    def parse(self, opts: list[str]): ...

    @interface
    def debug(self): ...

    @classmethod
    @interface
    def help(cls) -> str: ...


_TaskT = TypeVar("_TaskT", bound=TaskBase)


def new(cls: type[_TaskT], opts: list[str]) -> _TaskT:
    obj = cls.__new__(cls)
    obj.parse(opts)
    obj.__init__()
    return obj


class Static(TaskBase):
    @classmethod
    @interface
    def parse(cls, opts: list[str]): ...

    @classmethod
    @interface
    def debug(cls): ...

    def __new__(cls):
        return cls

    def __init__(): ...


class Unique(TaskBase): ...  # TODO
