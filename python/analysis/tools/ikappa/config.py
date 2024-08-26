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
    log_height = 50
    side_width = 200
    background = "#E8E8E8"
