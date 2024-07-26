import logging
from collections import defaultdict
from functools import cache
from io import StringIO
from pathlib import Path
from typing import overload

import fsspec

_SCHEME = ":##"
_KEY = "@@"


class DeserializationError(Exception):
    __module__ = Exception.__module__

    def __init__(self, msg):
        self.msg = msg


def _mapping_scheme(arg: str):
    arg = arg.split(_SCHEME, 1)
    if len(arg) == 1:
        return None, arg[0]
    else:
        return arg[0], arg[1]


def _mapping_nested_keys(arg: str):
    arg = arg.rsplit(_KEY, 1)
    if len(arg) == 1:
        return arg[0], None
    else:
        return arg[0], arg[1].split(".")


def _deserialize(data: str, protocol: str):
    match protocol:
        case "yaml":
            import yaml

            return yaml.safe_load(data)
        case "json":
            import json

            return json.loads(data)
        case "py":
            import importlib

            mods = data.split(".")
            try:
                mod = importlib.import_module(".".join(mods[:-1]))
                return vars(getattr(mod, mods[-1]))
            except Exception:
                raise DeserializationError(f'Failed to import "{data}"')
        case "csv":
            import pandas as pd

            return pd.read_csv(StringIO(data)).to_dict(orient="list")
        case _:
            raise DeserializationError(f'Unsupported protocol "{protocol}"')


@cache
def _deserialize_file(path: str, formatter: str):
    suffix = Path(path).suffix
    match suffix:
        case ".yml":
            protocol = "yaml"
        case ".yaml" | ".json" | ".csv":
            protocol = suffix[1:]
        case _:
            raise DeserializationError(f'Unsupported file "{path}"')
    try:
        with fsspec.open(path, "rt") as f:
            data = f.read()
            if formatter is not None:
                data = data.format(**mapping(formatter))
            return _deserialize(data, protocol)
    except Exception:
        raise DeserializationError(f'Failed to read file "{path}"')


def escape(obj) -> str:
    if not isinstance(obj, str):
        import json

        obj = f"json{_SCHEME}{json.dumps(obj)}"
    return obj


def mapping(arg: str, default: str = "yaml", formatter: str = None):
    """
    - `{data}`: parse as yaml
    - `yaml:##{data}`: parse as yaml
    - `json:##{data}`: parse as json
    - `csv:##{data}`: parse as csv
    - `file:##{path}`: read from file, support .yaml(.yml), .json .csv
    - `py:##{module.class}`: parse as python import

    `file`, `py` support an optional suffix `@@{key}.{key}...` to select a nested dict
    """
    if arg is None:
        return None
    if arg == "":
        return {}

    def warn(msg: str):
        logging.warning(f'{msg} when parsing "{arg}"')

    protocol, data = _mapping_scheme(arg)
    if protocol is None:
        protocol = default
    keys = None
    if protocol in ("file", "py"):
        data, keys = _mapping_nested_keys(data)
    try:
        if protocol == "file":
            result = _deserialize_file(data, formatter)
        else:
            result = _deserialize(data, protocol)
    except DeserializationError as e:
        warn(e.msg)
        return
    if keys is not None:
        for i, k in enumerate(keys):
            try:
                result = result[k]
            except Exception:
                warn(f'Failed to select key "{".".join(keys[:i+1])}"')
                return
    return result


def _split_with_empty(text: str, sep: str):
    if text == "":
        return []
    return text.split(sep)


@overload
def grouped_mappings(
    opts: list[list[str]], sep: str
) -> dict[frozenset[str], list[str]]: ...
@overload
def grouped_mappings(
    opts: list[list[str]], sep: None = None
) -> dict[str, list[str]]: ...
def grouped_mappings(opts: list[list[str]], sep: str = None):
    result = defaultdict(list)
    for opt in opts:
        if len(opt) < 2:
            continue
        else:
            arg = opt[0]
            if sep is not None:
                arg = frozenset(_split_with_empty(arg, sep))
            result[arg].extend(opt[1:])
    return result
