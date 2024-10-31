from ...typetools import dict_proxy
from ...utils import import_
from ._dict import mapping


def instance(__opt: list[str], __pkg: str, **__kwargs):
    if len(__opt) < 1:
        return None
    parts = f"{__pkg}.{__opt[0]}".split(".")
    _, cls = import_(".".join(parts[:-1]), parts[-1])
    if cls is None:
        return None
    if len(__opt) > 1:
        dict_proxy(__kwargs).update(*map(mapping, __opt[1:]))
    return cls(**__kwargs)
