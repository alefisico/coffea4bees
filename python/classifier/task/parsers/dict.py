import logging
from pathlib import Path
from typing import Any, Mapping


def _parse_scheme(opt: str):
    opt = opt.split(':///', 1)
    if len(opt) == 1:
        return None, opt[0]
    else:
        return opt[0], opt[1]


def parse_dict(opt: str) -> Mapping[str, Any]:
    '''
    - `{data}`: parse as yaml
    - `yaml:///{data}`: parse as yaml
    - `json:///{data}`: parse as json
    - `py:///{module.class}`: parse as python import
    - `file:///{path}`: read from file, support .yaml/.yml, .json
    '''
    protocol, data = _parse_scheme(opt)
    if protocol == 'file':
        match Path(data).suffix:
            case '.yaml' | '.yml':
                protocol = 'yaml'
            case '.json':
                protocol = 'json'
            case _:
                logging.error(f'Unsupported file: {data}')
                return
        try:
            import fsspec
            with fsspec.open(data, 'rt') as f:
                data = f.read()
        except:
            logging.error(f'Failed to read file: {data}')
            return
    match protocol:
        case None | 'yaml':
            import yaml
            return yaml.safe_load(data)
        case 'json':
            import json
            return json.loads(data)
        case 'py':
            import importlib
            mods = data.split('.')
            try:
                mod = importlib.import_module('.'.join(mods[:-1]))
                return vars(getattr(mod, mods[-1]))
            except:
                logging.error(f'Failed to import {data}')
        case _:
            logging.error(f'Unsupported protocol: {protocol}')
