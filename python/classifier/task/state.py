from __future__ import annotations

import logging
from textwrap import indent
from typing import Any, Mapping

from ..process import status
from ..typetools import dict_proxy
from .special import Static

_MAX_WIDTH = 30


def _is_private(name: str):
    return name.startswith("_")


def _is_special(name: str):
    return name.startswith("__") and name.endswith("__")


def _is_state(var: tuple[str, Any]):
    name, value = var
    return not _is_special(name) and not isinstance(value, classmethod)


class _CachedStateMeta(type):
    __cached_states__ = {}

    def __getattribute__(self, __name: str):
        if not _is_private(__name):
            value = vars(self).get(__name)
            parser = vars(self).get(f"_{__name}")
            if isinstance(parser, classmethod):
                if __name not in self.__cached_states__:
                    self.__cached_states__[__name] = parser.__func__(self, value)
                return self.__cached_states__[__name]
        return super().__getattribute__(__name)


class GlobalState(metaclass=_CachedStateMeta):
    _states: list[type[GlobalState]] = []

    def __init_subclass__(cls):
        cls._states.append(cls)


class _share_global_state:
    def __getstate__(self):
        return (
            *filter(
                lambda x: x[1],
                (
                    (cls, dict(filter(_is_state, vars(cls).items())))
                    for cls in GlobalState._states
                ),
            ),
        )

    def __setstate__(self, states: tuple[tuple[type[GlobalState], dict[str]]]):
        self._states = states

    def __call__(self):
        for cls, vars in self._states:
            for k, v in vars.items():
                setattr(cls, k, v)
        del self._states


status.initializer.add_unique(_share_global_state)


class Cascade(GlobalState, Static):
    @classmethod
    def __mod_name__(cls):
        return ".".join(f"{cls.__module__}.{cls.__name__}".split(".")[3:])

    @classmethod
    def parse(cls, opts: list[str]):
        from . import parse

        proxy = dict_proxy(cls)
        for opt in opts:
            data = parse.mapping(opt)
            if isinstance(data, Mapping):
                proxy.update(dict(filter(_is_state, data.items())))
            else:
                logging.error(
                    f"Unsupported data {data} when updating {cls.__name__}, expect a mapping."
                )

    @classmethod
    def help(cls):
        from base_class.typetools import get_partial_type_hints, type_name
        from rich.markup import escape

        from .task import _INDENT

        try:
            annotations = get_partial_type_hints(cls, include_extras=True)
        except Exception:
            annotations = cls.__annotations__
        keys = dict(filter(_is_state, vars(cls).items()))
        infos = [f"usage: {cls.__mod_name__()} OPTIONS [OPTIONS ...]", ""]
        if cls.__doc__:
            doc = filter(None, (l.strip() for l in cls.__doc__.split("\n")))
            infos.extend([*doc, ""])
        infos.append("options:")
        for k, v in keys.items():
            info = k
            if k in annotations:
                info += f": [green]{escape(type_name(annotations[k]))}[/green]"
            value = str(v)
            truncate = False
            if "\n" in value:
                value = value.split("\n", 1)[0]
                truncate = True
            if len(value) > _MAX_WIDTH:
                value = value[:_MAX_WIDTH]
                truncate = True
            info += f' = {value}{"..." if truncate else ""}'
            infos.append(indent(info, _INDENT))
        infos.append("")
        return "\n".join(infos)
