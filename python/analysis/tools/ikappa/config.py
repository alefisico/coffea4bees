from types import MappingProxyType
from typing import Callable

Datasets: dict[str, tuple[str, ...]] = MappingProxyType(
    {
        "ggF": (r"^GluGluToHHTo4B_cHHH(?P<kl>[p\d]+)$",),
    }
)

Stacks: tuple[tuple[str, tuple[str, ...]]] = (
    (
        "Background",
        (
            "Multijet",
            "TTToSemiLeptonic",
            "TTToHadronic",
            "TTTo2L2Nu",
        ),
    ),
)
XRootD: tuple[str, ...] = (
    "root://cmseos.fnal.gov//store/",
    "root://eosuser.cern.ch//eos/",
)

CouplingScan: dict[str, tuple[float, float, float]] = MappingProxyType(
    {
        "kl": (-5, 10, 0.1),
        None: (0, 5, 0.1),
    }
)

Palette: tuple[str, ...] = (
    "#DC143C",
    "#FF8C00",
    "#FFD700",
    "#32CD32",
    "#00BFFF",
    "#4169E1",
    "#8A2BE2",
    "#FF1493",
    "#A0522D",
    "#808000",
    "#228B22",
    "#D2B48C",
    "#66CDAA",
    "#EE82EE",
    "#808080",
)


class UI:
    height_log = 50
    height_multichoice = 40
    height_figure = 400
    width_side = 200
    width_numeric_input = 80

    color_background = "#E8E8E8"
    color_border = "#C8C8C8"
    border = f"1px solid {color_border}"


FloatFormat: Callable[[float], str] = "{:.6g}".format
