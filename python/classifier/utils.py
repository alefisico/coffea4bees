import importlib
import logging
from typing import Callable, Iterable, TypeVar

_GroupT = TypeVar("_GroupT", bound=Iterable)


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


class noop:
    def __getattr__(self, _):
        return self

    def __call__(self, *_, **__):
        return self


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
