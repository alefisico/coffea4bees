from collections import defaultdict

from hist import Hist


def sanitized(hists: dict[str, Hist], categories: set[str]):
    cats = sorted(categories)
    groups: dict[tuple, list[str]] = defaultdict(list)
    for k, v in hists.items():
        bins = {
            axis.name: (
                axis._ax.traits_overflow,
                axis._ax.traits_underflow,
                *axis,
            )
            for axis in v.axes
            if axis.name in categories
        }
        if len(bins) < len(cats):
            groups[None].append(k)
        groups[tuple(bins[cat] for cat in cats)].append(k)
    return sorted(groups.values(), key=len, reverse=True)
