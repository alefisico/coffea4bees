from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.task import Cascade

if TYPE_CHECKING:
    from base_class.system.eos import EOS, PathLike


class IO(Cascade):
    output: EOS = "."

    @classmethod
    def get__output(cls, value: PathLike):
        from base_class.system.eos import EOS

        return EOS(value).mkdir(recursive=True)


class Scheduler(Cascade):
    socket_timeout: float = None
    max_retry: int = 2

    @classmethod
    def set__socket_timeout(cls, value: float):
        import socket

        socket.setdefaulttimeout(value)


class DataLoader(Cascade):
    batch_io: int = 1_000_000
    batch_eval: int = 2**15
    num_workers: int = 0
