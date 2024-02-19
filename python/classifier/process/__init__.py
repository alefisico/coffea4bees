from __future__ import annotations

import logging
import os
from multiprocessing import current_process
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext
    from multiprocessing.process import BaseProcess

    class Context(BaseContext):
        Process: type[BaseProcess]


def is_main():
    return current_process().name == 'MainProcess'


def is_poxis():
    return os.name == 'posix'


def n_cpu():
    return os.cpu_count()


def get_context(
    method: Literal['fork', 'forkserver', 'spawn'] = ...,
    library: Literal['torch', 'builtins'] = 'torch',
    preload: list[str] = None,
) -> Context:
    if method is ...:
        method = 'forkserver' if is_poxis() else 'spawn'
    if not is_poxis() and method.startswith('fork'):
        logging.warn(
            f'"{method}" is not supported on non-posix systems, fallback to "spawn"')
        method = 'spawn'
    if method == 'fork':
        logging.warn(
            f'"{method}" is unsafe, consider using "spawn" or "forkserver" instead')
    match library:
        case 'builtins':
            import multiprocessing as mp
        case 'torch':
            import torch.multiprocessing as mp
        case _:
            raise ValueError(f'Unknown library "{library}"')
    ctx = mp.get_context(method)
    if method == 'forkserver' and preload is not None:
        ctx.set_forkserver_preload(preload)
    return ctx


class default:
    context: Context = None
    initializer: Callable = None
