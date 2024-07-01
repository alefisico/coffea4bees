from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import os
import socket
from dataclasses import dataclass
from enum import Flag
from functools import wraps
from threading import Lock
from typing import Any, Callable, Concatenate, NamedTuple, ParamSpec, TypeVar, overload
from uuid import uuid4

import fsspec

from ..config import setting as cfg
from ..typetools import Method
from . import is_poxis
from .connection import Client, Packet, Server
from .initializer import status

__all__ = [
    "Monitor",
    "Reporter",
    "Proxy",
    "post",
    "connect_to_monitor",
]


class Node(NamedTuple):
    ip: str
    pid: int


def _get_host():
    return socket.gethostbyname(socket.gethostname())


class _start_reporter:
    def __getstate__(self):
        match _Status.now():
            case _Status.Monitor:
                return Monitor.current()._address
            case _Status.Reporter:
                return Reporter.current()._address

    def __setstate__(self, address: str):
        self._address = address

    def __call__(self):
        Reporter.init(self._address)


class _Status(Flag):
    Unknown = 0b000
    Monitor = 0b100
    Reporter = 0b010
    Fresh = 0b001

    @classmethod
    def now(cls):
        status = _Status.Unknown
        if Monitor.current() is not None:
            status |= cls.Monitor
        if Reporter.current() is not None:
            status |= cls.Reporter
        if status == _Status.Unknown:
            status |= cls.Fresh
        return status


_SingletonT = TypeVar("_SingletonT", bound="_Singleton")


class _Singleton:
    __allowed_process__ = _Status.Fresh | _Status.Monitor

    def __init_subclass__(cls) -> None:
        cls.__instance = None

    def __new__(cls, *_, **__):
        if cls.__instance is not None:
            raise RuntimeError(f"{cls.__name__} is already initialized")
        if _Status.now() not in cls.__allowed_process__:
            name = str(cls.__allowed_process__).removeprefix(_Status.__name__ + ".")
            raise RuntimeError(f"{cls.__name__} must be initialized in {name} process")
        cls.__instance = super().__new__(cls)
        return cls.__instance

    @classmethod
    def init(cls: type[_SingletonT], *args, **kwargs) -> _SingletonT:
        if cls.__instance is None:
            cls.__instance = cls(*args, **kwargs)
        return cls.__instance

    @classmethod
    def current(cls: type[_SingletonT]) -> _SingletonT:
        return cls.__instance

    @classmethod
    def reset(cls):
        cls.__instance = None


@dataclass
class _Packet(Packet):
    def __post_init__(self):
        self.retry = self.retry or cfg.Monitor.retry_max
        super().__post_init__()


class _post:
    def __init__(self, func, wait: bool, retry: int):
        wraps(func)(self)
        self._func = func
        self._name = func.__name__
        self._wait = wait
        self._retry = retry

    def __call__(self, cls: type[Proxy], *args, **kwargs):
        packet = _Packet(
            cls.init,
            self._name,
            args,
            kwargs,
            wait=self._wait,
            lock=self._wait,
            retry=self._retry,
        )
        match _Status.now():
            case _Status.Monitor:
                if self._wait:
                    return packet()
                else:
                    return Monitor.current()._jobs.put(packet)
            case _Status.Reporter:
                return Reporter.current().send(packet)


_CallbackP = ParamSpec("_CallbackP")
_CallbackReturnT = TypeVar("_CallbackReturnT")


@overload
def post(
    func: Callable[Concatenate[Any, _CallbackP], _CallbackReturnT], /
) -> Method[_CallbackP, _CallbackReturnT]: ...
@overload
def post(
    wait_for_return: bool = False,
    max_retry: int = None,
) -> Callable[
    [Callable[Concatenate[Any, _CallbackP], _CallbackReturnT]],
    Method[_CallbackP, _CallbackReturnT],
]: ...
def post(
    func=None,
    *,
    wait_for_return: bool = False,
    max_retry: int = None,
):
    if func is None:
        return lambda func: post(
            func,
            wait_for_return=wait_for_return,
            max_retry=max_retry,
        )
    else:
        return classmethod(
            _post(
                func,
                wait=wait_for_return,
                retry=max_retry,
            )
        )


class Monitor(Server, _Singleton):
    __allowed_process__ = _Status.Fresh

    def __init__(self):
        # address
        _, port = cfg.Monitor.address
        if port is None:
            uuid = f"monitor-{uuid4()}"
            address = f"/tmp/{uuid}" if is_poxis() else rf"\\.\pipe\{uuid}"
        else:
            address = (_get_host(), port)
        super().__init__(address=address)

    def _start(self):
        return super().start()

    def _stop(self):
        return super().stop()

    @classmethod
    def start(cls):
        self = cls.init()
        if self._start():
            status.initializer.add_unique(_start_reporter)

    @classmethod
    def stop(cls):
        cls.init()._stop()

    @classmethod
    def lock(cls):
        return cls.current()._lock


class Reporter(Client, _Singleton):
    __allowed_process__ = _Status.Fresh

    def __init__(self, address: tuple[str, int | None]):
        self.reconnect_delay = cfg.Monitor.reconnect_delay
        if address[1] is None:
            address = address[0]
        super().__init__(address)
        Recorder.register(Recorder.name())

    def _stop(self):
        return super().stop()

    @classmethod
    def stop(cls):
        cls.current()._stop()


class _ProxyMeta(type):
    def __getattr__(cls: type[Proxy], name: str):
        return getattr(cls.init(), name)


class Proxy(_Singleton, metaclass=_ProxyMeta):
    _lock: Lock = None

    @classmethod
    def lock(cls):
        if _Status.now() != _Status.Monitor:
            raise RuntimeError("lock can only be accessed in monitor process")
        if cls._lock is None:
            cls._lock = Lock()
        return cls._lock


class Recorder(Proxy):
    _node = (_get_host(), os.getpid())
    _name = f"{_node[0]}/pid-{_node[1]}/{mp.current_process().name}"

    _reporters: dict[str, str]
    _data: list[tuple[str, Callable[[], bytes]]]

    def __init__(self):
        self._reporters = {self._name: "main"}
        self._data = [(cfg.Monitor.file, Recorder.serialize)]

    @post
    def register(self, name: str):
        index = f"#{len(self._reporters)}"
        self._reporters[name] = index
        logging.info(f'"{name}" is registered as [repr.number]\[{index}][/repr.number]')
        return index

    @classmethod
    def name(cls):
        return cls._name

    @classmethod
    def node(cls) -> Node:
        return cls._node

    @classmethod
    def registered(cls, name: str):
        index = cls._reporters.get(name)
        if index is None:
            index = cls.register(name)
        return index

    @classmethod
    def to_dump(cls, file: str, func: Callable[[], bytes]):
        cls._data.append((file, func))

    @classmethod
    def serialize(cls):
        import json

        return json.dumps(cls._reporters, indent=4).encode()

    @classmethod
    def dump(cls):
        if (_Status.now() == _Status.Monitor) and (not cfg.IO.monitor.is_null):
            for file, func in cls._data:
                if file is not None:
                    with fsspec.open(cfg.IO.monitor / file, "wb") as f:
                        f.write(func())


def connect_to_monitor():
    Reporter.init(cfg.Monitor.address)
    status.initializer.add_unique(_start_reporter)
    atexit.register(Reporter.current().stop)


def wait_for_monitor():
    if _Status.now() is _Status.Monitor:
        Monitor.current().stop()
