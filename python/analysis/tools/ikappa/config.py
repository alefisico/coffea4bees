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

CouplingScan = MappingProxyType(
    {
        "kl": (-5, 10, 0.1),
        None: (0, 5, 0.1),
    }
)


class UI:
    height_log = 50
    height_multichoice = 40
    width_side = 200
    width_numeric_input = 80

    color_background = "#E8E8E8"
    border = "1px solid #C8C8C8"
