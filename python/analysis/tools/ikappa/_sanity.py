from collections import defaultdict

from hist import Hist

from ._bh import BHAxis


def group_by_categories(
    hists: dict[str, Hist], categories: set[str]
) -> tuple[list[str], dict[str, list[Hist]]]:
    cats = sorted(categories)
    groups: dict[tuple, list[str]] = defaultdict(list)
    bhaxis = BHAxis(flow=True)
    for k, v in hists.items():
        bins = {
            axis.name: (
                bhaxis.flow(axis),
                *axis,
            )
            for axis in v.axes
            if axis.name in categories
        }
        if len(bins) < len(cats):
            groups[None].append(k)
        groups[tuple(bins[cat] for cat in cats)].append(k)
    return cats, dict(sorted(groups.items(), key=lambda x: len(x[1]), reverse=True))


def group_to_str(group: tuple) -> str:
    underflow, overflow, *bins = group
    lines = []
    if underflow:
        lines.append("[underflow]")
    if overflow:
        lines.append("[overflow]")
    lines.append(", ".join(str(b) for b in bins))
    return " ".join(lines)


def sanitized(hists: dict[str, Hist], categories: set[str]):
    _, groups = group_by_categories(hists, categories)
    groups.pop(None, None)
    return list(groups.values())
