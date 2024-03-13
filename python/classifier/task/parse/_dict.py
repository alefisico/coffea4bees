import logging
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import overload

from ._utils import fsspec_read


def _mapping_scheme(arg: str):
    arg = arg.split(":///", 1)
    if len(arg) == 1:
        return None, arg[0]
    else:
        return arg[0], arg[1]


def _mapping_nested_keys(arg: str):
    arg = arg.rsplit("@@", 1)
    if len(arg) == 1:
        return arg[0], None
    else:
        return arg[0], arg[1].split(".")


def mapping(arg: str):
    """
    - `{data}`: parse as yaml
    - `yaml:///{data}`: parse as yaml
    - `json:///{data}`: parse as json
    - `csv:///{data}`: parse as csv
    - `file:///{path}`: read from file, support .yaml(.yml), .json .csv
    - `py:///{module.class}`: parse as python import

    `file`, `py` support an optional suffix `@@{key}.{key}...` to select a nested dict
    """
    if arg is None:
        return None
    if arg == "":
        return {}

    def error(msg: str):
        logging.error(f'{msg} when parsing "{arg}"')

    protocol, data = _mapping_scheme(arg)
    keys = None
    if protocol in ("file", "py"):
        data, keys = _mapping_nested_keys(data)
    if protocol == "file":
        suffix = Path(data).suffix
        match suffix:
            case ".yml":
                protocol = "yaml"
            case ".yaml" | ".json" | ".csv":
                protocol = suffix[1:]
            case _:
                error(f'Unsupported file "{data}"')
                return
        try:
            data = fsspec_read(data)
        except:
            error(f'Failed to read file "{data}"')
            return

    result = None
    match protocol:
        case None | "yaml":
            import yaml

            result = yaml.safe_load(data)
        case "json":
            import json

            result = json.loads(data)
        case "py":
            import importlib

            mods = data.split(".")
            try:
                mod = importlib.import_module(".".join(mods[:-1]))
                result = vars(getattr(mod, mods[-1]))
            except:
                error(f'Failed to import "{data}"')
        case "csv":
            import pandas as pd

            result = pd.read_csv(StringIO(data)).to_dict(orient="list")
        case _:
            error(f'Unsupported protocol "{protocol}"')

    if result is not None and keys is not None:
        for i, k in enumerate(keys):
            try:
                result = result[k]
            except:
                error(f'Failed to select key "{".".join(keys[:i+1])}"')
                return
    return result


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
                arg = frozenset(arg.split(sep))
            result[arg].extend(opt[1:])
    return result
