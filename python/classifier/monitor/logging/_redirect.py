from __future__ import annotations

import logging
from typing import Iterable

from ...config.setting import Monitor as cfg
from ...config.state import RepoInfo
from ...process.monitor import Recorder, callback
from ..backends import Platform


class MultiPlatformLogRecord(logging.LogRecord):
    msg: str | Platform

    def getMessage(self):
        return self.msg


class MultiPlatformHandler(logging.Handler):
    __instance: MultiPlatformHandler = None

    def __init__(
        self, level: int | str = 0, handlers: Iterable[logging.Handler] = None
    ):
        super().__init__(level)
        self._handlers = [*(handlers or ())]

    @classmethod
    def init(cls, **kwargs):
        if cls.__instance is None:
            cls.__instance = cls(**kwargs)
        return cls.__instance

    def emit(self, record: logging.LogRecord):
        if cfg.log_enable:
            record.__class__ = MultiPlatformLogRecord
            record.name = Recorder.name()
            record.pathname = RepoInfo.get_url(record.pathname)
            self._emit(record)

    @callback(max_retry=1)
    def _emit(self, record: MultiPlatformLogRecord):
        record.name = Recorder.registered(record.name)
        for handler in self._handlers:
            if hasattr(handler, "__platform__") and isinstance(record.msg, Platform):
                if handler.__platform__ not in record.msg:
                    continue
            handler.handle(record)
