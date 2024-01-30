from __future__ import annotations

from functools import cached_property

from .utils import unique


class Label:
    def __init__(self, name: str, *subs: Label):
        self.name = name
        self.subs = subs
        self.base: type[LabelCollection] = None
        self.field: str = None
        self._index: int = None

    @cached_property
    def index(self) -> list[int]:
        if self._index is None:
            return sorted(unique(ref.index for ref in self.subs))
        return [self._index]

    @cached_property
    def label(self) -> dict[str, bool]:
        return {label.field: label._index in self.index for label in self.base.all}

    @cached_property
    def fields(self) -> list[str]:
        return [label.field for label in self.base.all if label._index in self.index]

    def __repr__(self):
        if self.subs:
            return f'{self.field}{self.subs}'
        return self.field

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Label):
            return self.field == __value.field
        return False

    def __hash__(self) -> int:
        return hash(self.field)


class LabelCollectionMeta(type):
    all: list[Label]

    def __repr__(cls) -> str:
        return f'{cls.__name__}{cls.all}'


class LabelCollection(metaclass=LabelCollectionMeta):
    def __init_subclass__(cls):
        cls.all = []
        for name, value in vars(cls).items():
            if isinstance(value, Label):
                value.base = cls
                value.field = name
                if not value.subs:
                    value._index = len(cls.all)
                    cls.all.append(value)

    @classmethod
    @property
    def fields(cls) -> list[str]:
        return [label.field for label in cls.all]
