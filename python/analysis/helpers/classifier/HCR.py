import awkward as ak
import base_class.awkward as akext
import numpy as np
from base_class.root import Chunk, Friend
from base_class.system.eos import PathLike


def _build_cutflow(*selections):
    if len(selections) == 1:
        return selections[0]
    selection = akext.to.numpy(selections[0], copy=True).astype(bool)
    for s in selections[1:]:
        selection[selection] = s
    return ak.Array(selection)


def dump_friend(
    events: ak.Array,
    output: PathLike,
    name: str,
    data: ak.Array,
    dump_naming: str = "{path1}/{name}_{uuid}_{start}_{stop}_{path0}",
):
    chunk = Chunk.from_coffea_events(events)
    friend = Friend(name)
    friend.add(chunk, data)
    friend.dump(output, dump_naming)
    return {name: friend}


def dump_input_friend(
    events: ak.Array,
    output: PathLike,
    name: str,
    *selections: ak.Array,
    CanJet: str = "canJet",
    NotCanJet: str = "notCanJet",
    weight: str = "weight",
    dump_naming: str = "{path1}/{name}_{uuid}_{start}_{stop}_{path0}",
):
    selection = _build_cutflow(*selections)
    padded = akext.pad.selected()
    data = ak.Array(
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
    )
    return dump_friend(
        events=events,
        output=output,
        name=name,
        data=data,
        dump_naming=dump_naming,
    )


def dump_JCM_weight(
    events: ak.Array,
    output: PathLike,
    name: str,
    *selections: ak.Array,
    pseudo_tag: str = "pseudoTagWeight",
    dump_naming: str = "{path1}/{name}_{uuid}_{start}_{stop}_{path0}",
):
    if not pseudo_tag in events.fields:
        weight = np.ones(len(selections[0]), dtype=np.float64)
    else:
        selection = _build_cutflow(*selections)
        padded = akext.pad.selected(1)
        weight = padded(events[pseudo_tag], selection)
    return dump_friend(
        events=events,
        output=output,
        name=name,
        data={"pseudoTagWeight": weight},
        dump_naming=dump_naming,
    )
