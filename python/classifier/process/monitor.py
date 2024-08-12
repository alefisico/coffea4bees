from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import os
import socket
import time
from dataclasses import dataclass
from enum import Flag
from functools import wraps
from multiprocessing.connection import Client, Connection, Listener
from queue import PriorityQueue
from threading import Lock, Thread
from typing import Any, Callable, Concatenate, NamedTuple, ParamSpec, TypeVar, overload
from uuid import uuid4

import fsspec

from ..config import setting as cfg
from ..typetools import Method
from ..utils import noop
from . import is_poxis
from .initializer import status

__all__ = [
    "MonitorError",
    "Monitor",
    "Reporter",
    "Proxy",
    "post",
    "connect_to_monitor",
]


class Node(NamedTuple):
    ip: str
    pid: int


def _close_connection(connection: Connection):
    try:
        connection.close()
    except:
        pass


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
class _Packet:
    cls: type[Proxy] = None
    func: str = None
    args: tuple = None
    kwargs: dict = None
    wait: bool = None
    lock: bool = None
    retry: int = None

    def __post_init__(self):
        self._timestamp = time.time_ns()
        self._n_retried = 0
        self.retry = self.retry or cfg.Monitor.retry_max

    def __call__(self):
        lock = self.cls.lock() if self.lock else noop
        try:
            with lock:
                return getattr(self.cls, self.func)._func(
                    self.cls.init(), *self.args, **self.kwargs
                )
        except Exception as e:
            logging.error(e, exc_info=e)

    def __lt__(self, other):
        if isinstance(other, _Packet):
            if self.cls is None:
                return False
            elif other.cls is None:
                return True
            return (
                self._n_retried,
                self._timestamp,
            ) < (
                other._n_retried,
                other._timestamp,
            )
        return NotImplemented


class _post:
    def __init__(self, func, wait: bool, retry: int):
        wraps(func)(self)
        self._func = func
        self._name = func.__name__
        self._wait = wait
        self._retry = retry

    def __call__(self, cls: type[Proxy], *args, **kwargs):
        packet = _Packet(
            cls,
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


class MonitorError(Exception):
    __module__ = Exception.__module__


class Monitor(_Singleton):
    __allowed_process__ = _Status.Fresh

    def __init__(self):
        # states
        self._accepting = False
        self._lock = Lock()

        # listener
        _, port = cfg.Monitor.address
        if port is None:
            uuid = f"monitor-{uuid4()}"
            self._address = f"/tmp/{uuid}" if is_poxis() else rf"\\.\pipe\{uuid}"
        else:
            self._address = (_get_host(), port)
        self._listener: tuple[Listener, Thread] = None
        self._runner: Thread = None

        # clients
        self._jobs: PriorityQueue[_Packet] = PriorityQueue()
        self._connections: list[Connection] = []
        self._handlers: list[Thread] = []

    @classmethod
    def start(cls):
        self = cls.init()
        if self._listener is None:
            self._listener = (
                Listener(self._address),
                Thread(target=self._listen, daemon=True),
            )
            self._listener[1].start()
            self._runner = Thread(target=self._run, daemon=True)
            self._runner.start()
            status.initializer.add_unique(_start_reporter)

    @classmethod
    def stop(cls):
        self = cls.init()
        if self._listener is not None:
            # close listener
            listener, thread = self._listener
            self._listener = None
            if self._accepting:
                try:
                    Client(self._address).close()
                except ConnectionRefusedError:
                    pass
            listener.close()
            thread.join()
            # close runner
            self._jobs.put(_Packet())
            self._runner.join()
            # close connections
            for connection in self._connections:
                _close_connection(connection)
            self._connections.clear()
            for handler in self._handlers:
                handler.join()
            self._handlers.clear()

    def _run(self):
        while (packet := self._jobs.get()).cls is not None:
            packet()

    def _listen(self):
        while True:
            try:
                self._accepting = True
                connection = self._listener[0].accept()
                self._accepting = False
                self._connections.append(connection)
                if self._listener is not None:
                    handler = Thread(
                        target=self._handle, args=(connection,), daemon=True
                    )
                    self._handlers.append(handler)
                    handler.start()
                else:
                    _close_connection(connection)
                    break
            except OSError:
                break
            finally:
                self._accepting = False

    def _handle(self, connection: Connection):
        while True:
            try:
                packet: _Packet = connection.recv()
                if packet.wait:
                    connection.send(packet())
                else:
                    packet._n_retried = 0
                    self._jobs.put(packet)
            except:
                _close_connection(connection)
                break

    @classmethod
    def lock(cls):
        return cls.current()._lock


class Reporter(_Singleton):
    __allowed_process__ = _Status.Fresh

    def __init__(self, address: tuple[str, int | None]):
        if address[1] is None:
            self._address = address[0]
        else:
            self._address = address

        self._lock = Lock()
        self._jobs: PriorityQueue[_Packet] = PriorityQueue()
        self._sender: Connection = None
        self._thread = Thread(target=self._send_non_blocking, daemon=True)
        self._thread.start()
        Recorder.register(Recorder.name())

    def _send(self, packet: _Packet):
        with self._lock:
            try:
                if self._sender is None:
                    self._sender = Client(self._address)
                self._sender.send(packet)
                if packet.wait:
                    return self._sender.recv()
            except Exception as e:
                if self._sender is not None:
                    _close_connection(self._sender)
                    self._sender = None
                if isinstance(e, ConnectionError):
                    raise
                raise MonitorError

    def _send_non_blocking(self):
        while (packet := self._jobs.get()).cls is not None:
            packet._n_retried += 1
            try:
                self._send(packet)
            except (MonitorError, ConnectionError) as e:
                if isinstance(e, ConnectionError):
                    if self._thread is None:
                        return
                    time.sleep(cfg.Monitor.reconnect_delay)
                if packet._n_retried < packet.retry:
                    self._jobs.put(packet)

    def send(self, packet: _Packet):
        if packet.wait:
            return self._send(packet)
        else:
            self._jobs.put(packet)

    def send_atexit(self):
        self.stop()
        if not self._jobs.empty():
            self._jobs.put(_Packet())
            self._send_non_blocking()

    @classmethod
    def stop(cls):
        self = cls.current()
        if self._thread is not None:
            with self._lock:
                running = self._sender is not None
            if running:
                self._jobs.put(_Packet())
                thread = self._thread
                self._thread = None
                thread.join()


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
    atexit.register(Reporter.current().send_atexit)


def wait_for_monitor():
    if _Status.now() is _Status.Monitor:
        Monitor.current().stop()
