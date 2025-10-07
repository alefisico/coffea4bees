from __future__ import annotations

from collections import defaultdict
from operator import add
from typing import TYPE_CHECKING, Any, Callable

import dask

from src.dask.functions import reduction

if TYPE_CHECKING:
    from hist import Hist

    from ._analysis import AnalysisOutput


def _reveal_key(outputs: list[AnalysisOutput], key: str):
    for output in outputs:
        if isinstance(output, dict) and "result" in output:
            output = output["result"]
        else:
            continue
        for k, v in output.items():
            if isinstance(v, dict) and key in v:
                yield k, v[key]


def dump_hists(
    outputs: list[AnalysisOutput],
    dumper: Callable[[Any], Any],
    split_every=None,
):
    collected = defaultdict(list)
    categories = None
    for _, v in _reveal_key(outputs, "hists"):
        hists: dict[str, Hist] = v["hists"]
        if categories is None:
            categories = v["categories"]
        elif categories != v["categories"]:
            raise ValueError("All histograms must have the same categories")
        for name, hist in hists.items():
            collected[name].append(hist)
    return dask.delayed(dumper)(
        {
            "hists": {
                name: reduction(add, *hists, split_every=split_every)
                for name, hists in collected.items()
            },
            "categories": categories,
        }
    )
