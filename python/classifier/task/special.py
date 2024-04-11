from __future__ import annotations

from typing import TypeVar

__all__ = ["interface", "new", "TaskBase", "Static", "Unique"]

_InterfaceT = TypeVar("_InterfaceT")


class InterfaceError(NotImplementedError):
    def __init__(self, owner, func):
        import inspect

        signature = str(inspect.signature(func)).replace("'", "").replace('"', "")
        super().__init__(
            f"Interface is not implemented: {owner.__name__}.{func.__name__}{signature}"
        )


class _Interface:
    def __init__(self, func):
        self.func = func

    def __get__(self, _, owner):
        raise InterfaceError(owner, self.func)


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


class WorkInProgress: ...
