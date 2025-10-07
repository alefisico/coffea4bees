from __future__ import annotations

import awkward as ak
import src.dask.awkward as dakext
import numpy as np


class TightBTagJCM:
    def __init__(self, weights: list[float], start: int = 4):
        self._weights = np.ones(start + len(weights), dtype=float)
        self._weights[start:] = weights

    @dakext.partition_mapping
    def __call__(self, n_jets: ak.Array):
        return ak.from_numpy(self._weights[dakext.safe.to_numpy(n_jets)])
