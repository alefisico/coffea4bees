from types import MappingProxyType

Datasets = MappingProxyType(
    {
        "ggF": (r"GluGluToHHTo4B_cHHH(?P<kl>[p\d]+)",),
    }
)

Stacks = (
    (
        "Multijet",
        "TTToSemiLeptonic",
        "TTToHadronic",
        "TTTo2L2Nu",
    ),
)

XRootD = (
    "root://cmseos.fnal.gov//store/",
    "root://eosuser.cern.ch//eos/",
)


class UI:
    height_log = 50
    height_select_bar = 40
    width_side = 200
    color_background = "#E8E8E8"
