from __future__ import annotations

import logging

import tblib.pickling_support

from ...config.setting import Monitor as cfg
from ...process.monitor import Recorder
from ._redirect import MultiPlatformHandler


def _common():
    tblib.pickling_support.install()


def _disable_logging():
    logging.basicConfig(handlers=[logging.NullHandler()], level=None)


def setup_reporter():
    if cfg.log_enable:
        _common()
        return logging.basicConfig(
            handlers=[MultiPlatformHandler(level=cfg.logging_level)],
            level=cfg.logging_level,
        )
    _disable_logging()


def setup_monitor():
    if cfg.log_enable:
        _common()
        handlers = []
        if cfg.console_enable:
            from ..backends.console import Dashboard as _CD
            from ._console import ConsoleDump, ConsoleHandler

            ConsoleDump.init()
            Recorder.to_dump(cfg.file_log, ConsoleDump.serialize)
            handlers.append(ConsoleDump.handler)
            handlers.append(ConsoleHandler.new(_CD.console))
        if handlers:
            return logging.basicConfig(
                handlers=[
                    MultiPlatformHandler.init(
                        level=cfg.logging_level, handlers=handlers
                    )
                ],
                level=cfg.logging_level,
            )
    _disable_logging()
