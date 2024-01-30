import os
from multiprocessing import current_process
from multiprocessing.process import BaseProcess
from multiprocessing.context import BaseContext
from typing import Literal

from .monitor import console
from .utils import type_hint_only


@type_hint_only
class Context(BaseContext):
    Process: type[BaseProcess]


def is_main():
    return current_process().name == 'MainProcess'


def is_poxis():
    return os.name == 'posix'


def get_context(
        method: Literal['fork', 'spawn', 'forkserver'] = ...,
        library: Literal['builtin', 'torch'] = 'builtin',
        preload: list[str] = None) -> Context:
    if method is ...:
        method = 'forkserver' if is_poxis() else 'spawn'
    if not is_poxis() and method.startswith('fork'):
        console.warn(
            f'"{method}" is not supported on non-posix systems, fallback to "spawn"')
        method = 'spawn'
    if method == 'fork':
        console.warn(
            f'"{method}" is unsafe, consider using "spawn" or "forkserver" instead')
    match library:
        case 'builtin':
            import multiprocessing as mp
        case 'torch':
            import torch.multiprocessing as mp
        case _:
            raise ValueError(f'Unknown library "{library}"')
    ctx = mp.get_context(method)
    if method == 'forkserver' and preload is not None:
        ctx.set_forkserver_preload(preload)
    return ctx
