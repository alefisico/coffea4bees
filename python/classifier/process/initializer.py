from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

from ..task.state import share_global_state

if TYPE_CHECKING:
    from . import Context


class _initializer:
    def __init__(self, *funcs: Callable):
        self.funcs: list[Callable] = []
        self.add(
            inherit_context_initializer(),
            share_global_state(),
        )
        self.add(*funcs)

    def add(self, *funcs: Callable):
        self.funcs.extend(funcs)

    def __call__(self):
        for func in self.funcs:
            func()


class inherit_context_initializer:
    def __getstate__(self):
        return (status.context, status.initializer)

    def __setstate__(self, states: tuple[Context, _initializer]):
        self._context, self._initializer = states

    def __call__(self):
        status.context = self._context
        status.initializer = self._initializer
        status.is_main = False


class torch_set_sharing_strategy:
    def __init__(
        self, strategy: Literal["file_system", "file_descriptor"] = "file_system"
    ):
        import torch.multiprocessing as mp

        strategies = mp.get_all_sharing_strategies()
        if strategy not in strategies:
            raise ValueError(
                f'Unknown strategy "{strategy}", available strategies are {strategies}'
            )
        self.strategy = strategy

    def __call__(self):
        import torch.multiprocessing as mp

        mp.set_sharing_strategy(self.strategy)


class status:
    context: Context = None
    initializer = _initializer()
    is_main: bool = True
