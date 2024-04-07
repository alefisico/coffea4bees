from __future__ import annotations

import importlib
import logging
from fractions import Fraction
from functools import cache
from typing import Callable, Iterable, TypeVar

_GroupT = TypeVar("_GroupT", bound=Iterable)
_ItemT = TypeVar("_ItemT")


def _subgroup(group: list, left: int, remain: int):
    if remain == 0:
        yield group
    else:
        size = len(group)
        for i in range(size, left, -1):
            yield from _subgroup(group[: i - 1] + group[i:], i - 1, remain - 1)


def subgroups(group: _GroupT, new: Callable[[Iterable], _GroupT] = None):
    if new is None:
        new = type(group)
    group = [*group]
    for i in range(len(group)):
        for sub in _subgroup(group, 0, i):
            yield new(sub)
    yield new(())


@cache
def _subsets_cached(group):
    return tuple(subgroups(group, frozenset))


def subsets(group: frozenset[_ItemT]) -> frozenset[frozenset[_ItemT]]:
    return _subsets_cached(group)


class NOOP:
    def __getattr__(self, _):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __repr__(self):
        return "no-op"


class _NoopMeta(NOOP, type): ...


class noop(metaclass=_NoopMeta):
    def __new__(cls, *_, **__):
        return cls


def import_(modname: str, clsname: str):
    _mod, _cls = None, None
    try:
        _mod = importlib.import_module(modname)
    except ModuleNotFoundError:
        ...
    except Exception as e:
        logging.error(e)
    if _mod is not None and clsname != "*":
        try:
            _cls = getattr(_mod, clsname)
        except AttributeError:
            ...
        except Exception as e:
            logging.error(e)
    return _mod, _cls


def append_unique_instance(collection: list[_ItemT], item: _ItemT | type[_ItemT]):
    if isinstance(item, type):
        for i in collection:
            if isinstance(i, item):
                return collection
        item = item()
    elif item in collection:
        return collection
    collection.append(item)
    return collection


def keep_fraction(fraction: Fraction, indices):
    return (indices % fraction.denominator) < fraction.numerator
