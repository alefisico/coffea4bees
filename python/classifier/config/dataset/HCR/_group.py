from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    from classifier.df.tools import DFProcessor

    ProcessorGenerator = Callable[[frozenset[str]], Iterable[DFProcessor]]


class fullmatch:
    def __init__(
        self,
        *groups: Iterable[str],
        processors: Iterable[DFProcessor],
        name: str = None,
    ):
        self._gs = (*map(frozenset, groups),)
        self._ps = (*processors,)
        self.name = name

    def __call__(self, groups: frozenset[str]):
        if any(g <= groups for g in self._gs):
            yield from self._ps


@dataclass
class _regex:
    pattern: str

    _unique = False

    def __post_init__(self):
        self._pattern = re.compile(self.pattern)

    def any(self, matches: tuple[re.Match]):
        yield from ()

    def none(self):
        yield from ()

    def __call__(self, groups: frozenset[str]):
        matched = (*filter(None, map(self._pattern.fullmatch, groups)),)
        if len(matched) >= 1:
            if self._unique and len(matched) > 1:
                raise ValueError(f'Multiple "{self.pattern}" matched in {groups}')
            yield from self.any(matched)
        else:
            yield from self.none()


@dataclass
class regex(_regex):
    processors: Iterable[DFProcessor]
    default: Iterable[DFProcessor] = ()

    def any(self, _):
        yield from self.processors

    def none(self):
        yield from self.default


@dataclass
class add_column(_regex):
    key: str
    default: Any = None
    dtype: type = float

    _unique = True

    def any(self, matches: tuple[re.Match, ...]):
        from classifier.df.tools import add_columns

        yield add_columns(**{self.key: self.dtype(matches[0].group(self.key))})

    def none(self):
        if self.default is not None:
            from classifier.df.tools import add_columns

            yield add_columns(**{self.key: self.default})


@dataclass
class add_year(add_column):
    key: str = field(init=False, default="year")
    pattern: str = field(init=False, default=r"year:\w*(?P<year>\d{2}).*")
    default: None = field(init=False, default=None)
    dtype: type = field(init=False, default=int)


@dataclass
class add_single_label(_regex):
    key: str = field(init=False, default="label")
    pattern: str = field(init=False, default=r"label:(?P<label>.*)")

    _unique = True

    def any(self, matches: tuple[re.Match, ...]):
        from classifier.df.tools import add_label_index

        yield add_label_index(matches[0].group(self.key))
