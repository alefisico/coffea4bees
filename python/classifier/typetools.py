import builtins
from enum import Enum
from functools import partial
from typing import Mapping, MutableMapping


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
