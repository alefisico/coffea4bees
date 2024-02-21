from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

from . import default
from .state import share_global_state

if TYPE_CHECKING:
    from . import Context


class DefaultInitializer:
    def __init__(self, *funcs: Callable):
        self.funcs: list[Callable] = []
        self.add(share_global_state())
        self.add(*funcs)

    def add(self, *funcs: Callable):
        self.funcs.extend(funcs)

    def __call__(self):
        for func in self.funcs:
            func()


class inherit_context_initializer:
    def __init__(self, context: Context = None, initializer: Callable = None):
        self._context = context or default.context
        self._initializer = initializer or default.initializer

    def __call__(self):
        default.context = self._context
        default.initializer = self._initializer


class torch_set_sharing_strategy:
    def __init__(self, strategy: Literal['file_system', 'file_descriptor'] = 'file_system'):
        import torch.multiprocessing as mp
        strategies = mp.get_all_sharing_strategies()
        if strategy not in strategies:
            raise ValueError(
                f'Unknown strategy "{strategy}", available strategies are {strategies}')
        self.strategy = strategy

    def __call__(self):
        import torch.multiprocessing as mp
        mp.set_sharing_strategy(self.strategy)
