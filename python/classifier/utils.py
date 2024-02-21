import logging
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


def parse_intervals(ranges: list[str], max: int) -> list[int]:
    result = []
    for r in ranges:
        rs = r.split('-')
        try:
            match len(rs):
                case 1:
                    result.append(int(rs[0]))
                case 2:
                    result.extend(range(int(rs[0]), int(rs[1])+1))
                case _:
                    raise ValueError
        except ValueError:
            logging.error(f'Invalid range {r}')
    return sorted(filter(lambda x: 0 <= x < max, set(result)))
