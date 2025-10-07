import logging

import dask
import dask_awkward as dak
from dask.delayed import optimize as delayed_optimize


def _extended_delayed_optimize(dsk, keys, **_):
    dsk = dak.optimize.all_optimizations(dsk, keys)
    dsk = delayed_optimize(dsk, keys)
    return dsk


def setup():
    """
    - setup dask logging levels
    - add dask-awkward optimizations to delayed
    """
    dask.config.set(delayed_optimize=_extended_delayed_optimize)
    for level, loggers in {
        "ERROR": (
            "distributed",
            "distributed.core",
            "distributed.worker",
            "distributed.scheduler",
        ),
        "CRITICAL": (
            "distributed.nanny",
            "bokeh",
            "tornado",
            "tornado.application",
        ),
    }.items():
        for logger in loggers:
            logging.getLogger(logger).setLevel(level)
