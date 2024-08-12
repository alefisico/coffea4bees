from __future__ import annotations

import logging

import tblib.pickling_support

from ...config.setting import monitor as cfg
from ...process.monitor import Recorder
from ._redirect import MultiPlatformHandler


def _common():
    tblib.pickling_support.install()


def _disable():
    logging.basicConfig(handlers=[logging.NullHandler()], level=None)


@cfg.check(cfg.Log, default=_disable, is_callable=True)
def setup_reporter():
    _common()
    return logging.basicConfig(
        handlers=[MultiPlatformHandler()],
        level=cfg.Log.level,
    )


@cfg.check(cfg.Log, default=_disable, is_callable=True)
def setup_monitor():
    _common()
    handlers = []
    if cfg.Console.enable:
        from ..backends.console import Dashboard as _CD
        from ._console import ConsoleDump, ConsoleHandler

        ConsoleDump.init()
        Recorder.to_dump(cfg.Log.file, ConsoleDump.serialize)
        handlers.append(ConsoleDump.handler)
        handlers.append(ConsoleHandler.new(_CD.console))
    if handlers:
        return logging.basicConfig(
            handlers=[MultiPlatformHandler.init(handlers=handlers)],
            level=cfg.Log.level,
        )
