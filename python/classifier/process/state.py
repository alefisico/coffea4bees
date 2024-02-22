from __future__ import annotations

import logging
from typing import Any, Mapping, get_type_hints

from ..task.parsers import parse_dict


def _is_special(name: str):
    return name.startswith('__') and name.endswith('__')


def _is_state(var: tuple[str, Any]):
    name, value = var
    return not _is_special(name) and not isinstance(value, classmethod)


class GlobalState:
    _states: list[type[GlobalState]] = []

    def __init_subclass__(cls):
        cls._states.append(cls)


class share_global_state:
    def __getstate__(self):
        return [(cls, dict(filter(_is_state, vars(cls).items())))
                for cls in GlobalState._states]

    def __setstate__(self, states: list[tuple[type[GlobalState], dict]]):
        self._states = states

    def __call__(self):
        for cls, vars in self._states:
            for k, v in vars.items():
                setattr(cls, k, v)
        del self._states


class Cascade(GlobalState):
    @classmethod
    def parse(cls, opts: list[str]):
        for opt in opts:
            data = parse_dict(opt)
            if isinstance(data, Mapping):
                for k, v in data.items():
                    if not _is_special(k):
                        setattr(cls, k, v)
            else:
                logging.error(
                    f'Unsupported data {data} when updating {cls.__name__}, expect a mapping.')

    @classmethod
    def help(cls):
        import os

        from base_class.typetools import type_name
        from rich.markup import escape

        _MAX_WIDTH = os.get_terminal_size().columns // 3

        annotations = get_type_hints(cls)
        keys = dict(filter(_is_state, vars(cls).items()))
        infos = []
        if cls.__doc__:
            infos.append(cls.__doc__)
        for k, v in keys.items():
            info = k
            if k in annotations:
                info += f': [green]{escape(type_name(annotations[k]))}[/green]'
            value = str(v)
            truncate = False
            if '\n' in value:
                value = value.split('\n', 1)[0]
                truncate = True
            if len(value) > _MAX_WIDTH:
                value = value[:_MAX_WIDTH]
                truncate = True
            info += f' = {value}{"..." if truncate else ""}'
            infos.append(info)
        return '\n'.join(infos)
