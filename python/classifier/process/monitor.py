from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from enum import Flag
from functools import wraps
from multiprocessing.connection import Client, Connection, Listener
from queue import PriorityQueue
from threading import Lock, Thread
from typing import Any, Callable, Concatenate, ParamSpec, TypeVar, overload
from uuid import uuid4

from ..config.setting import Monitor as Setting
from ..typetools import Method
from ..utils import noop
from . import is_poxis
from .initializer import status

__all__ = [
    "MonitorError",
    "Monitor",
    "Reporter",
    "Proxy",
    "callback",
    "connect_to_monitor",
]


def _close_connection(connection: Connection):
    try:
        connection.close()
    except:
        pass


def _get_host():
    return socket.gethostbyname(socket.gethostname())


def _parse_url(url: str):
    host, port = url.rsplit(":", 1)
    return host, int(port)


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
    cls: type[Proxy]
    func: str
    args: tuple
    kwargs: dict
    wait: bool
    lock: bool

    def __post_init__(self):
        self.retry = Setting.max_retry

    def __call__(self):
        lock = self.cls.lock() if self.lock else noop
        try:
            with lock:
                return getattr(self.cls, self.func)(*self.args, **self.kwargs)
        except Exception as e:
            logging.error(e)

    def __lt__(self, other):
        if isinstance(other, _Packet):
            return self.retry > other.retry
        return NotImplemented


class _callback:
    def __init__(self, func, wait: bool, lock: bool):
        wraps(func)(self)
        self._func = func
        self._name = func.__name__
        self._wait = wait
        self._lock = lock

    def __call__(self, cls: type[Proxy], *args, **kwargs):
        match _Status.now():
            case _Status.Monitor:
                return self._func(cls.init(), *args, **kwargs)
            case _Status.Reporter:
                return Reporter.current().send(
                    _Packet(
                        cls,
                        self._name,
                        args,
                        kwargs,
                        wait=self._wait,
                        lock=self._lock,
                    )
                )


_CallbackP = ParamSpec("_CallbackP")
_CallbackReturnT = TypeVar("_CallbackReturnT")


@overload
def callback(
    func: Callable[Concatenate[Any, _CallbackP], _CallbackReturnT], /
) -> Method[_CallbackP, _CallbackReturnT]: ...
@overload
def callback(
    func: None = None, wait_for_return: bool = False, acquire_class_lock: bool = True
) -> Callable[
    [Callable[Concatenate[Any, _CallbackP], _CallbackReturnT]],
    Method[_CallbackP, _CallbackReturnT],
]: ...
def callback(
    func=None,
    wait_for_return: bool = False,
    acquire_class_lock: bool = True,
):
    if func is None:
        return lambda func: callback(func, wait_for_return, acquire_class_lock)
    else:
        return classmethod(_callback(func, wait_for_return, acquire_class_lock))


class MonitorError(Exception):
    __module__ = Exception.__module__


class Monitor(_Singleton):
    __allowed_process__ = _Status.Fresh

    def __init__(self):
        # states
        self._accepting = False
        self._lock = Lock()

        # listener
        if Setting.port is None:
            uuid = f"monitor-{uuid4()}"
            self._address = f"/tmp/{uuid}" if is_poxis() else rf"\\.\pipe\{uuid}"
        else:
            self._address = (_get_host(), Setting.port)
        self._listener: tuple[Listener, Thread] = None

        # clients
        self._connections: list[Connection] = []
        self._handlers: list[Thread] = []
        self._reporters: dict[str, int] = {}

    @classmethod
    def start(cls, standalone=False):
        self = cls.init()
        if self._listener is None:
            self._listener = (
                Listener(self._address),
                Thread(target=self._listen, daemon=(not standalone)),
            )
            self._listener[1].start()
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
            # close connections
            for connection in self._connections:
                _close_connection(connection)
            self._connections.clear()
            for handler in self._handlers:
                handler.join()
            self._handlers.clear()

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
                runner: _Packet = connection.recv()
                result = runner()
                if runner.wait:
                    connection.send(result)
            except:
                _close_connection(connection)
                break

    @classmethod
    def lock(cls):
        return cls.current()._lock

    @callback
    def register(cls, name: str):
        index = len(cls.current()._reporters)
        cls.current()._reporters[name] = index
        logging.info(f'"{name}" is registered as \[#{index}]')

    @classmethod
    def registered(cls, name: str):
        total = len(str(len(cls.current()._reporters)))
        index = cls.current()._reporters.get(name)
        if index is None:
            index = "?" * (total + 1)
        else:
            index = f"#{index:0{total}d}"
        return index


class Reporter(_Singleton):
    __allowed_process__ = _Status.Fresh

    def __init__(self, address: str):
        self._address = address
        self._name = (
            f"{_get_host()}/pid-{os.getpid()}/{status.context.current_process().name}"
        )

        self._lock = Lock()
        self._jobs: PriorityQueue[_Packet] = PriorityQueue()
        self._sender: Connection = None
        self._thread = Thread(target=self._send_non_blocking, daemon=True)
        self._thread.start()
        Monitor.register(self._name)

    def _send(self, packet: _Packet):
        with self._lock:
            try:
                if self._sender is None:
                    self._sender = Client(self._address)
                self._sender.send(packet)
                if packet.wait:
                    return self._sender.recv()
            except:
                if self._sender is not None:
                    _close_connection(self._sender)
                    self._sender = None
                raise MonitorError

    def _send_non_blocking(self):
        while (packet := self._jobs.get()) is not None:
            packet.retry -= 1
            try:
                self._send(packet)
            except MonitorError:
                if packet.retry >= 1:
                    self._jobs.put(packet)

    def send(self, packet: _Packet):
        if packet.wait:
            return self._send(packet)
        else:
            self._jobs.put(packet)

    @classmethod
    def name(cls):
        return cls.current()._name

    @classmethod
    def stop(cls):
        cls.current()._jobs.put(None)
        cls.current()._thread.join()


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


def connect_to_monitor(address: str):
    Reporter.init(_parse_url(address))
    status.initializer.add_unique(_start_reporter)
