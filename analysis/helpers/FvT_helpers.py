from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import numpy as np
import awkward as ak
if TYPE_CHECKING:
    from .classifier.HCR import HCREnsemble

def compute_FvT(events, mask, **models: HCREnsemble):
    masked_events = events[mask]

    for name, model in models.items():
        if model is None:
            continue

        if name in events.fields:
            events[f"old_{name}"] = events[name]

        tmp_c_score, tmp_q_score = model(masked_events)

        c_score = np.ones((len(events),tmp_c_score.shape[1]))
        c_score[mask] = tmp_c_score
        q_score = np.ones((len(events),tmp_q_score.shape[1]))
        q_score[mask] = tmp_q_score

        del tmp_c_score, tmp_q_score

        classes = model.classes
        pd4 = c_score[:, classes.index("d4")]
        pd3 = c_score[:, classes.index("d3")]
        pt4 = c_score[:, classes.index("t4")]
        pt3 = c_score[:, classes.index("t3")]
        FvT = (pd4 - pt4) / pd3 # pt3 is not used anywhere
        FvT = np.where(np.isclose(FvT, 0), 1, FvT)

        events[name] = ak.zip({
            "pd4": pd4,
            "pd3": pd3,
            "pt4": pt4,
            "pt3": pt3,
            "FvT": FvT,
            "q_1234": q_score[:, 0],
            "q_1324": q_score[:, 1],
            "q_1423": q_score[:, 2],
        })

        # if f"old_{name}" in events.fields:
        #     print(events[f"old_{name}"].FvT[:10])
        #     print(events[name].FvT[:10])
        #     print("\n\n\n")
        #     error = ~np.isclose(events[f"old_{name}"].FvT, events[name].FvT, atol=1e-5, rtol=1e-3)
        #     if np.any(error):
        #         delta = np.abs(events[f"old_{name}"].FvT - events[name].FvT)
        #         worst = np.max(delta) == delta
        #         worst_events = events[worst][0]

        #         print( f"WARNING: Calculated {name} does not agree within tolerance for some events ({np.sum(error)}/{len(error)}) {delta[worst]} \n\n" )