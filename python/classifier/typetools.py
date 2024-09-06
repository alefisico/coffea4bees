import builtins
from _thread import LockType
from enum import Enum
from functools import partial
from threading import Lock
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Mapping,
    MutableMapping,
    ParamSpec,
    Protocol,
    TypeVar,
    runtime_checkable,
)
from uuid import uuid4

_MethodP = ParamSpec("_MethodP")
_MethodReturnT = TypeVar("_MethodReturnT")


class Method(Protocol, Generic[_MethodP, _MethodReturnT]):
    def __get__(
        self, instance: Any, owner: type | None = None
    ) -> Callable[_MethodP, _MethodReturnT]: ...
    def __call__(
        self_, self: Any, *args: _MethodP.args, **kwargs: _MethodP.kwargs
    ) -> _MethodReturnT: ...


class WithUUID:
    def __init__(self):
        super().__init__()
        self.uuid = uuid4()


class PicklableLock:
    def __init__(self):
        super().__init__()
        self.lock = Lock()

    def __copy__(self):
        new = self.__new__(self.__class__)
        new.__dict__ = self.__dict__.copy()
        return new

    def __getstate__(self):
        return self.__dict__ | {"lock": isinstance(self.lock, LockType)}

    def __setstate__(self, state):
        self.__dict__ = state
        self.lock = Lock() if self.lock else None


class dict_proxy(MutableMapping):
    def __init__(self, obj):
        self._object = obj
        if isinstance(obj, Mapping):
            self._mapping = obj
        else:
            self._mapping = obj.__dict__

        for k in ("set", "del"):
            if hasattr(self._mapping, f"__{k}item__"):
                func = getattr(self._mapping, f"__{k}item__")
            else:
                func = partial(getattr(builtins, f"{k}attr"), self._object)
            setattr(self, f"_{k}", func)

    def __getitem__(self, __key):
        return self._mapping.__getitem__(__key)

    def __iter__(self):
        return self._mapping.__iter__()

    def __len__(self):
        return len(self._mapping)

    def __delitem__(self, __key):
        return self._del(__key)

    def __setitem__(self, __key, __value):
        return self._set(__key, __value)

    def update(self, *mappings: Mapping):
        for mapping in mappings:
            for k, v in mapping.items():
                if k in self and isinstance(v, Mapping):
                    dict_proxy(self[k]).update(v)
                else:
                    self[k] = v
        return self


def enum_dict(enum: type[Enum]):
    return {i.name: i.value for i in enum}


@runtime_checkable
class FilenameProtocol(Protocol):
    def __filename__(self) -> str: ...


def filename(obj: Any) -> str:
    if isinstance(obj, FilenameProtocol):
        return obj.__filename__()
    elif isinstance(obj, Mapping):
        name = []
        for k, v in obj.items():
            name.append(f"{filename(k)}_{filename(v)}")
        return "__".join(name)
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, Iterable):
        return "-".join(map(filename, obj))
    else:
        return repr(obj)
