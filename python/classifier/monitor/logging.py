import logging

from rich.console import Console
from rich.logging import RichHandler

from ..config.setting import Monitor as Setting
from ..config.state import RepoInfo
from ..process import status
from ..process.monitor import Monitor, Reporter, callback
from ._rich_log_render import replace_log_render


class RemoteHandler(logging.Handler):
    console: Console = None

    @classmethod
    def init(cls):
        return cls

    def emit(self, record: logging.LogRecord):
        record.name = Reporter.name()
        record.pathname = RepoInfo.get_url(record.pathname)
        self.send(record)

    @callback(acquire_class_lock=False, max_retry=1)
    def send(self, record: logging.LogRecord):
        record.name = Monitor.registered(record.name)
        logging.getLogger().handle(record)


def setup_remote_logger():
    logging.basicConfig(handlers=[RemoteHandler()], level=Setting.logging_level)
    status.initializer.add_unique(setup_remote_logger)


def setup_main_logger():
    RemoteHandler.console = Console(record=True, markup=True)
    logging.basicConfig(
        handlers=[
            replace_log_render(
                RichHandler(markup=True, console=RemoteHandler.console),
                show_level=True,
                level_width=None,
            )
        ],
        level=Setting.logging_level,
        format="\[%(name)s] %(message)s",
    )
    status.initializer.add_unique(setup_remote_logger)
