import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from hashlib import sha1
from io import BytesIO
from typing import Any, Callable, Optional

import fsspec
import psutil
import yaml
from src.config._io import FileIORegistry
from src.storage.eos import EOS

_DEFAULT_TIME_FORMAT = "%Y-%m-%dT%H-%M-%S"


def _check_exe(p: Optional[psutil.Process]):
    if p is None:
        return None
    return p.exe()


def start_time(format: str = _DEFAULT_TIME_FORMAT):
    process = psutil.Process()
    exe = process.exe()
    while _check_exe(parent := process.parent()) == exe:
        process = parent
    return datetime.fromtimestamp(process.create_time()).strftime(format)


def current_time(format: str = _DEFAULT_TIME_FORMAT):
    return datetime.now().strftime(format)


def join_path(base: os.PathLike, *parts: str):
    return EOS(base).join(*parts)


@dataclass
class _SimpleStringIO:
    buffer: BytesIO
    encoding: str = "utf-8"

    def write(self, data: str):
        self.buffer.write(data.encode(self.encoding))


class _FileDumper(FileIORegistry[Callable[[BytesIO, Any], None]]):
    def __call__(self, obj: Any, url: str, use_buffer: bool = True):
        serializer, _, compression = self._get_handler(url)
        if use_buffer:
            buffer = BytesIO()
            serializer(buffer, obj)
        with fsspec.open(url, mode="wb", compression=compression) as f:
            if use_buffer:
                f.write(buffer.getvalue())
            else:
                serializer(f, obj)


@_FileDumper.register("json")
def json_deserializer(file: BytesIO, obj: Any):
    json.dump(obj, _SimpleStringIO(file))


@_FileDumper.register("yaml", "yml")
def yaml_deserializer(file: BytesIO, obj: Any):
    return yaml.safe_dump(obj, _SimpleStringIO(file))


@_FileDumper.register("pkl")
def pkl_deserializer(file: BytesIO, obj: Any):
    return pickle.dump(obj, file)


@_FileDumper.register("txt")
def write_string(file: BytesIO, obj: str):
    file.write(obj.encode("utf-8"))


@_FileDumper.register("bin")
def write_bytes(file: BytesIO, obj: bytes):
    file.write(obj)


dumps = _FileDumper()
"Dump a python object to a file."
dumper = partial(partial, dumps)
"Create a dumper from url."


def id_sha1(obj):
    "Generate a sha1 id for a python object."
    return sha1(pickle.dumps(obj)).hexdigest()
