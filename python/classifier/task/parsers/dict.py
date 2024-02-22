import logging
from pathlib import Path
from typing import Any, Mapping


def _parse_scheme(opt: str):
    opt = opt.split(':///', 1)
    if len(opt) == 1:
        return None, opt[0]
    else:
        return opt[0], opt[1]


def _parse_keys(opt: str):
    opt = opt.rsplit('=>', 1)
    if len(opt) == 1:
        return opt[0], None
    else:
        return opt[0], opt[1].split('.')


def parse_dict(opt: str) -> Mapping[str, Any]:
    '''
    - `{data}`: parse as yaml
    - `yaml:///{data}`: parse as yaml
    - `json:///{data}`: parse as json
    - `file:///{path}`: read from file, support .yaml(.yml), .json
    - `py:///{module.class}`: parse as python import

    `file`, `py` support an optional suffix `=>{key}.{key}...` to select a nested dict
    '''
    def error(msg: str):
        logging.error(f'{msg} when parsing "{opt}"')

    protocol, data = _parse_scheme(opt)
    keys = None
    if protocol in ('file', 'py'):
        data, keys = _parse_keys(data)
    if protocol == 'file':
        match Path(data).suffix:
            case '.yaml' | '.yml':
                protocol = 'yaml'
            case '.json':
                protocol = 'json'
            case _:
                error(f'Unsupported file "{data}"')
                return
        try:
            import fsspec
            with fsspec.open(data, 'rt') as f:
                data = f.read()
        except:
            error(f'Failed to read file "{data}"')
            return

    result = None
    match protocol:
        case None | 'yaml':
            import yaml
            result = yaml.safe_load(data)
        case 'json':
            import json
            result = json.loads(data)
        case 'py':
            import importlib
            mods = data.split('.')
            try:
                mod = importlib.import_module('.'.join(mods[:-1]))
                result = vars(getattr(mod, mods[-1]))
            except:
                error(f'Failed to import "{data}"')
        case _:
            error(f'Unsupported protocol "{protocol}"')

    if result is not None and keys is not None:
        for i, k in enumerate(keys):
            try:
                result = result[k]
            except:
                error(
                    f'Failed to select key "{".".join(keys[:i+1])}"')
                return
    return result
