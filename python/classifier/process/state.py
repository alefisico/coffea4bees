from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping


def _is_special(name: str):
    return name.startswith('__') and name.endswith('__')


def _is_state(var: tuple[str, Any]):
    name, value = var
    return not _is_special(name) and not isinstance(value, classmethod)


def _parse_scheme(opt: str):
    opt = opt.split(':///', 1)
    if len(opt) == 1:
        return None, opt[0]
    else:
        return opt[0], opt[1]


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
    def update(self, *opts: str):
        '''
         - `{data}`: parse as yaml
         - `yaml:///{data}`: parse as yaml
         - `json:///{data}`: parse as json
         - `py:///{module.class}`: parse as python import
         - `file:///{path}`: read from file, support .yaml/.yml, .json
        '''
        for opt in opts:
            protocol, data = _parse_scheme(opt)
            if protocol == 'file':
                path = Path(data)
                match path.suffix:
                    case '.yaml' | '.yml':
                        protocol = 'yaml'
                    case '.json':
                        protocol = 'json'
                    case _:
                        logging.error(f'Unsupported file: {path}')
                        continue
                try:
                    data = path.read_text()
                except:
                    logging.error(f'Failed to read file: {path}')
                    continue
            match protocol:
                case None | 'yaml':
                    import yaml
                    self.update_dict(yaml.safe_load(data))
                case 'json':
                    import json
                    self.update_dict(json.loads(data))
                case 'py':
                    import importlib
                    mods = data.split('.')
                    try:
                        mod = importlib.import_module('.'.join(mods[:-1]))
                        self.update_dict(vars(getattr(mod, mods[-1])))
                    except:
                        logging.error(f'Failed to import {data}')
                case _:
                    logging.error(f'Unsupported protocol: {protocol}')

    @classmethod
    def update_dict(cls, data):
        if isinstance(data, Mapping):
            for k, v in data.items():
                if not _is_special(k):
                    setattr(cls, k, v)
        else:
            logging.error(
                f'Unsupported data {data} when updating {cls.__name__}, expect a mapping.')
