from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from multiprocessing import connection as mpc
from queue import PriorityQueue
from threading import Lock, Thread
from typing import Callable, Protocol

from ..utils import noop


class _ClientError(Exception):
    __module__ = Exception.__module__


def _close_connection(connection: mpc.Connection):
    try:
        connection.close()
    except Exception:
        pass


class Proxy(Protocol):
    @classmethod
    def lock(self) -> Lock: ...


@dataclass
class Packet:
    obj: Callable[[], Proxy] = None
    func: str = ""
    args: tuple = ()
    kwargs: dict = None
    wait: bool = False
    lock: bool = False
    retry: int = 0

    def __post_init__(self):
        self._timestamp = time.time_ns()
        self._retried = 0
        self.kwargs = self.kwargs or {}

    def __call__(self):
        obj = self.obj()
        lock = obj.lock() if self.lock else noop
        try:
            with lock:
                return getattr(obj, self.func)._func(obj, *self.args, **self.kwargs)
        except Exception as e:
            logging.error(e, exc_info=e)

    def __lt__(self, other):
        if isinstance(other, Packet):
            if self.obj is None:
                return False
            elif other.obj is None:
                return True
            return (
                self._retried,
                self._timestamp,
            ) < (
                other._retried,
                other._timestamp,
            )
        return NotImplemented


class Server:
    def __init__(self, address):
        # address
        self._address = address

        # states
        self._accepting = False
        self._lock = Lock()

        # listener
        self._listener: tuple[mpc.Listener, Thread] = None
        self._runner: Thread = None

        # clients
        self._jobs: PriorityQueue[Packet] = PriorityQueue()
        self._connections: list[mpc.Connection] = []
        self._handlers: list[Thread] = []

    def start(self):
        if self._listener is None:
            self._listener = (
                mpc.Listener(self._address),
                Thread(target=self._listen, daemon=True),
            )
            self._listener[1].start()
            self._runner = Thread(target=self._run, daemon=True)
            self._runner.start()
            return True
        return False

    def stop(self):
        if self._listener is not None:
            # close listener
            listener, thread = self._listener
            self._listener = None
            if self._accepting:
                try:
                    mpc.Client(self._address).close()
                except ConnectionRefusedError:
                    pass
            listener.close()
            thread.join()
            # close runner
            self._jobs.put(Packet())
            self._runner.join()
            # close connections
            for connection in self._connections:
                _close_connection(connection)
            self._connections.clear()
            for handler in self._handlers:
                handler.join()
            self._handlers.clear()
            return True
        return False

    def _run(self):
        while (packet := self._jobs.get()).obj is not None:
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

    def _handle(self, connection: mpc.Connection):
        while True:
            try:
                packet: Packet = connection.recv()
                if packet.wait:
                    connection.send(packet())
                else:
                    packet._retried = 0
                    self._jobs.put(packet)
            except Exception:
                _close_connection(connection)
                break


class Client:
    reconnect_delay = 1  # seconds

    def __init__(self, address: str | tuple[str, int]):
        self._address = address
        self._lock = Lock()
        self._jobs: PriorityQueue[Packet] = PriorityQueue()
        self._sender: mpc.Connection = None
        self._thread = Thread(target=self._send_non_blocking, daemon=True)
        self._thread.start()

    def _send(self, packet: Packet):
        with self._lock:
            try:
                if self._sender is None:
                    self._sender = mpc.Client(self._address)
                self._sender.send(packet)
                if packet.wait:
                    return self._sender.recv()
            except Exception as e:
                if self._sender is not None:
                    _close_connection(self._sender)
                    self._sender = None
                if isinstance(e, ConnectionError):
                    raise
                raise _ClientError

    def _send_non_blocking(self):
        while (packet := self._jobs.get()).obj is not None:
            packet._retried += 1
            try:
                self._send(packet)
            except (_ClientError, ConnectionError) as e:
                if isinstance(e, ConnectionError):
                    if self._thread is None:
                        return
                    time.sleep(self.reconnect_delay)
                if packet._retried < packet.retry:
                    self._jobs.put(packet)

    def send(self, packet: Packet):
        if packet.wait:
            return self._send(packet)
        else:
            self._jobs.put(packet)

    def send_atexit(self):
        self.stop()
        if not self._jobs.empty():
            self._jobs.put(Packet())
            self._send_non_blocking()

    def stop(self):
        if self._thread is not None:
            with self._lock:
                running = self._sender is not None
            if running:
                self._jobs.put(Packet())
                thread = self._thread
                self._thread = None
                thread.join()
            return True
        return False
