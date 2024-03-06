import logging
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Iterable, overload


def _parse_scheme(opt: str):
    opt = opt.split(':///', 1)
    if len(opt) == 1:
        return None, opt[0]
    else:
        return opt[0], opt[1]


def _parse_keys(opt: str):
    opt = opt.rsplit('@@', 1)
    if len(opt) == 1:
        return opt[0], None
    else:
        return opt[0], opt[1].split('.')


def parse_dict(opt: str):
    '''
    - `{data}`: parse as yaml
    - `yaml:///{data}`: parse as yaml
    - `json:///{data}`: parse as json
    - `csv:///{data}`: parse as csv
    - `file:///{path}`: read from file, support .yaml(.yml), .json .csv
    - `py:///{module.class}`: parse as python import

    `file`, `py` support an optional suffix `@@{key}.{key}...` to select a nested dict
    '''
    def error(msg: str):
        logging.error(f'{msg} when parsing "{opt}"')

    protocol, data = _parse_scheme(opt)
    keys = None
    if protocol in ('file', 'py'):
        data, keys = _parse_keys(data)
    if protocol == 'file':
        suffix = Path(data).suffix
        match suffix:
            case '.yml':
                protocol = 'yaml'
            case '.yaml' | '.json' | '.csv':
                protocol = suffix[1:]
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
        case 'csv':
            import pandas as pd
            result = pd.read_csv(StringIO(data)).to_dict(orient='list')
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


@overload
def parse_group(opt: Iterable[tuple[str, str]], sep: str) -> dict[frozenset[str], list[str]]:
    ...


@overload
def parse_group(opt: Iterable[tuple[str, str]], sep: None = None) -> dict[str, list[str]]:
    ...


def parse_group(opt: Iterable[tuple[str, str]], sep: str = None):
    result = defaultdict(list)
    for k, v in opt:
        if sep is not None:
            k = frozenset(k.split(sep))
        result[k].append(v)
    return result
