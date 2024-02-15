import itertools
from operator import gt, lt
from types import ModuleType
from typing import Literal, TypeVar

from packaging import version

from .monitor import console


def version_warn(pkg: ModuleType, ver: str, op: Literal['<', '>']):
    ops = {'<': lt, '>': gt}
    warns = {'<': FutureWarning, '>': DeprecationWarning}
    if ops[op](version.parse(pkg.__version__), version.parse(ver)):
        console.warn(
            f'{pkg.__name__} version {op} {ver} may not work with this code, get {pkg.__version__}', warns[op])


def version_check(pkg: ModuleType, upper: str = None, lower: str = None):
    if upper is not None:
        version_warn(pkg, upper, '>')
    if lower is not None:
        version_warn(pkg, lower, '<')


class TypedStr:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'

    def __str__(self):
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, self.__class__):
            return self.name == __value.name
        return False


def ranges(seq: list[int]):
    for _, i in itertools.groupby(enumerate(seq), lambda x: x[1] - x[0]):
        i = list(i)
        yield i[0][1], i[-1][1]


def ranges_str(seq: list[int]):
    for a, b in ranges(seq):
        if a == b:
            yield str(a)
        else:
            yield f'{a}-{b}'


_TypeHintT = TypeVar('_TypeHintT')


def _type_hint_class(cls, *args, **kwargs):
    raise RuntimeError(f'{cls.__name__} is only used for type hinting')


def type_hint_only(cls: _TypeHintT) -> _TypeHintT:
    cls.__new__ = _type_hint_class
    return cls
