from typing import Callable, Generic, Iterable, TypeVar

__all__ = ['unique', 'SeqCall']

# PLAN add to base class

_SeqCallT = TypeVar('_SeqCallT')


def unique(seq: Iterable):
    return list(set(seq))


class SeqCall(Generic[_SeqCallT]):
    def __init__(self, *funcs: Callable[[_SeqCallT], _SeqCallT]):
        self.funcs = funcs

    def __call__(self, x) -> _SeqCallT:
        for func in self.funcs:
            x = func(x)
        return x
