import logging

from rich.logging import RichHandler

from ..config.setting import Monitor as Setting
from ..process import status
from ..process.monitor import Monitor, Reporter, callback


class RemoteHandler(logging.Handler):
    @classmethod
    def init(cls):
        return cls

    def emit(self, record: logging.LogRecord):
        record.name = Reporter.name()
        self.send(record)

    @callback(acquire_class_lock=False, max_retry=1)
    def send(self, record: logging.LogRecord):
        record.name = Monitor.registered(record.name)
        logging.getLogger().handle(record)


def setup_remote_logger():
    logging.basicConfig(handlers=[RemoteHandler()], level=Setting.logging_level)
    status.initializer.add_unique(setup_remote_logger)


def setup_main_logger():
    logging.basicConfig(
        handlers=[RichHandler(markup=True)],
        level=Setting.logging_level,
        format="\[%(name)s] %(message)s",
    )
    status.initializer.add_unique(setup_remote_logger)
