from types import ModuleType

from packaging import version


def version_check(pkg: ModuleType, upper: str = None, lower: str = None):
    current = version.parse(pkg.__version__)
    if upper is not None:
        upper = version.parse(upper)
        if current > upper:
            return False
    if lower is not None:
        lower = version.parse(lower)
        if current < lower:
            return False
    return True
