import awkward as ak
import base_class.awkward as akext
from base_class.root import Chunk, Friend
from base_class.system.eos import PathLike


def _build_cutflow(*selections):
    if len(selections) == 1:
        return selections[0]
    selection = akext.to.numpy(selections[0], copy=True).astype(bool)
    for s in selections[1:]:
        selection[selection] = s
    return ak.Array(selection)


def build_input_friend(
    events: ak.Array,
    output: PathLike,
    name: str,
    *selections: ak.Array,
    CanJet: str = "canJet",
    NotCanJet: str = "notCanJet",
    weight: str = "weight",
    dump_naming: str = "{path1}/{name}_{uuid}_{start}_{stop}_{path0}",
):
    chunk = Chunk.from_coffea_events(events)
    selection = _build_cutflow(*selections)
    padded = akext.pad.selected()
    friend = Friend(name)
    friend.add(
        chunk,
        ak.Array(
            {
                "CanJet": padded(
                    ak.zip(
                        {
                            "pt": events[CanJet].pt,
                            "eta": events[CanJet].eta,
                            "phi": events[CanJet].phi,
                            "mass": events[CanJet].mass,
                        }
                    ),
                    selection,
                ),
                "NotCanJet": padded(
                    ak.zip(
                        {
                            "pt": events[NotCanJet].pt,
                            "eta": events[NotCanJet].eta,
                            "phi": events[NotCanJet].phi,
                            "mass": events[NotCanJet].mass,
                            "isSelJet": events[NotCanJet].isSelJet,
                        }
                    ),
                    selection,
                ),
            }
            | akext.to.numpy(
                padded(
                    events[
                        [
                            "ZZSR",
                            "ZHSR",
                            "HHSR",
                            "SR",
                            "SB",
                            "fourTag",
                            "threeTag",
                            "passHLT",
                            "nSelJets",
                            "xbW",
                            "xW",
                        ]
                    ],
                    selection,
                )
            )
            | {"weight": padded(events[weight], selection)}
        ),
    )
    friend.dump(output, dump_naming)
    return {name: friend}
