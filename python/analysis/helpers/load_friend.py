# The module contains temporary functions to load new friend trees in a backward compatible manner
# TODO: remove in the future
import awkward as ak
from base_class.root import Chunk, Friend


def rename_FvT_friend(chunk: Chunk, friend: Friend):
    FvT = friend.arrays(chunk)
    fields = set(FvT.fields)
    renamed = {k: FvT[k] for k in ("FvT", "q_1234", "q_1324", "q_1423")}
    for k, v in (
        ("p_d4", "pd4"),
        ("p_d3", "pd3"),
        ("p_t4", "pt4"),
        ("p_t3", "pt3"),
        ("p_m4", "pm4"),
        ("p_m3", "pm3"),
        ("p_ttbar", "pt"),
    ):
        if k in fields:
            renamed[v] = FvT[k]
    return ak.zip(renamed)
