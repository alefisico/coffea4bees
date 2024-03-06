import awkward as ak
import base_class.awkward as akext
from base_class.root import Chunk, Friend
from base_class.system.eos import PathLike


def _build_cutflow(*selections):
    pad = akext.pad.selected(False)
    selection = selections[-1]
    for s in selections[-2::-1]:
        selection = s & pad(selection, s)
    return selection


def build_input_friend(
    events: ak.Array,
    output: PathLike,
    name: str,
    *selections: ak.Array,
    canJet: str = 'canJet',
    notCanJet: str = 'notCanJet',
    weight: str = 'weight',
):
    chunk = Chunk.from_coffea_events(events)
    selection = _build_cutflow(*selections)
    padded = akext.pad.selected()
    friend = Friend(name)
    friend.add(chunk, ak.Array({
        'CanJet': padded(ak.zip({
            'pt': events[canJet].pt,
            'eta': events[canJet].eta,
            'phi': events[canJet].phi,
            'mass': events[canJet].mass,
        }), selection),
        'NotCanJet': padded(ak.zip({
            'pt': events[notCanJet].pt,
            'eta': events[notCanJet].eta,
            'phi': events[notCanJet].phi,
            'mass': events[notCanJet].mass,
            'isSelJet': events[notCanJet].isSelJet,
        }), selection)
    } | akext.to.numpy(padded(
        events[[
            'ZZSR', 'ZHSR', 'HHSR', 'SR', 'SB',
            'fourTag', 'threeTag', 'passHLT',
            'event', 'nSelJets', weight,
        ]], selection))
    ))
    friend.dump(output)
    return {name: friend}
