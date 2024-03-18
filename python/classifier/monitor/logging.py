import logging

from rich.logging import RichHandler

from ..process import status


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(RichHandler(markup=True))
    status.initializer.add_unique(setup_logger)
