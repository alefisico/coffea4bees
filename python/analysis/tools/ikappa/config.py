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
    # plot
    figure_height = 400
    legend_width = 200
    legend_checkbox_width = 27
    legend_glyph_width = legend_checkbox_width * 3
    # browser
    log_height = 50
    sidebar_width = 200
    # widgets
    multichoice_height = 40
    numeric_input_width = 80

    background_color = "#E8E8E8"
    border_color = "#C8C8C8"
    border = f"1px solid {border_color}"


FloatingPrecision: int = 6
