raise NotImplementedError  # noqa # TODO work in progress
from __future__ import annotations

import logging
import os
from collections import defaultdict
from multiprocessing import Lock as Lock_
from multiprocessing.connection import Client, Connection, Listener
from multiprocessing.synchronize import Lock
from queue import Queue
from threading import Thread
from typing import Any, Callable

from . import Context, get_context, is_main, is_poxis
from ..static import Constant


class Monitor:
    # data
    _queue: defaultdict[str, Queue[tuple[str, tuple, dict[str, Any]]]]

    # connection
    _authkey: bytes

    _listener: Listener
    _listener_thread: Thread

    _worker: Worker

    # state
    _queue_lock: Lock
    _accepting: bool = False
    _stopped: bool = True

    @staticmethod
    def start():
        if not Monitor._stopped:
            logging.warn('Monitor is already running')
            return
        if is_main():
            Monitor._stopped = False
            Monitor._authkey = os.urandom(Constant.authkey_size)
            Monitor._queue = defaultdict(Queue)
            Monitor._listener = Listener(
                address=...,  # TODO Address class
                authkey=Monitor._authkey)
            Monitor._listener_thread = Thread(target=Monitor._listen)
            Monitor._listener_thread.start()
            Monitor._queue_lock = Lock_()

    @staticmethod
    def _handle(connection: Connection):
        while True:
            try:
                data = connection.recv()
                Monitor._add(data)
            except EOFError:
                break

    @staticmethod
    def _listen():
        while True:
            try:
                Monitor._accepting = True
                connection = Monitor._listener.accept()
                Monitor._accepting = False
                if not Monitor._stopped:
                    Thread(target=Monitor._handle, args=(connection,)).start()
                else:
                    connection.close()
                    break
            except OSError:
                break
            finally:
                Monitor._accepting = False

    @staticmethod
    def _add(data: tuple[str, str, tuple, dict[str, Any]]):
        if is_main():
            Monitor._queue_lock.acquire()
            Monitor._queue[data[0]].put(data[1:])
            Monitor._queue_lock.release()

    @staticmethod
    def stop():
        if is_main() and not Monitor._stopped:
            Monitor._stopped = True
            Monitor._listener.close()
            if is_poxis() and Monitor._accepting:
                # in POSIX, the Listener.accept() will not raise an OSError after the Listener.close()
                try:
                    Client(...,  # TODO Address class
                           authkey=Monitor._authkey).close()
                except ConnectionRefusedError:
                    pass
            Monitor._listener_thread.join()

    @staticmethod
    def send(cls: str, method: str, *args, **kwargs):
        if is_main():
            Monitor._add(cls, method, *args, **kwargs)
        else:
            if Monitor._worker is None:
                raise RuntimeError('no worker is available')
            Monitor._worker.send(cls, method, *args, **kwargs)

    def __init_subclass__(cls):
        return super().__init_subclass__()


class Worker:
    def __init__(
            self,
            address: str | tuple[str, int],  # TODO make Address class
            authkey: bytes):
        self._address = address  # TODO Address class
        self._authkey = authkey
        self._sender: Connection = None

    def send(self, cls: str, method: str, *args, **kwargs):
        try:
            if self._sender is None:
                self._sender = Client(
                    self._address, authkey=self._authkey)  # TODO Address class
            self._sender.send((cls, method, args, kwargs))
        except ConnectionRefusedError:
            if self._sender is not None:
                self._sender.close()
                self._sender = None

    @classmethod
    def _executor(
            cls,
            worker: Worker,
            target: Callable,
            args: tuple = None,
            kwargs: dict[str] = None):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        Monitor._worker = worker
        target(*args, **kwargs)

    @classmethod
    def Process(
            cls,
            target: Callable,
            args: tuple = None,
            kwargs: dict[str] = None,
            context: Context = None):
        worker = Worker(...,
                        Monitor._authkey)  # TODO Address class
        if context is None:
            context = get_context()
        process = context.Process(
            target=cls._executor,
            kwargs={
                'worker': worker,
                'target': target,
                'args': args,
                'kwargs': kwargs
            })
        return process

    ...  # TODO separate client and listener
    ...  # TODO spawn process and start worker
    ...  # TODO take authkey from Monitor
