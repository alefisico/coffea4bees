from types import MappingProxyType

# common xrootd paths
# used for autocompletion
XRootD: tuple[str, ...] = (
    "root://cmseos.fnal.gov//store/",
    "root://eosuser.cern.ch//eos/",
)

# common stacks
# used for initialization
# format: (name, (dataset1, dataset2, ...))
Stacks: tuple[tuple[str, tuple[str, ...]]] = (
    (
        "Background",
        (
            "TTTo2L2Nu",
            "TTToSemiLeptonic",
            "TTToHadronic",
            "QCD Multijet",
        ),
    ),
)

# possible dataset patterns for each model
# used for autocompletion and initialization
# format: {model: (pattern1, pattern2, ...)}
Datasets: dict[str, tuple[str, ...]] = MappingProxyType(
    {
        "ggF": (r"^GluGluToHHTo4B_cHHH(?P<kl>[p\d]+)$",),
    }
)

# default min, max and step for coupling sliders
# used for initialization
# format: {coupling: (min, max, step)}
CouplingScan: dict[str, tuple[float, float, float]] = MappingProxyType(
    {
        "kl": (-5, 10, 0.1),
        None: (0, 5, 0.1),  # all other couplings
    }
)

# plotting palette
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
)


class Plot:
    figure_height = 400  # px, height of the figure
    fill_alpha = 0.2  # transparency of the fill color, 0-1
    legend_width = 200  # px, width of the legend
    tooltip_float_precision = 4  # float precision for tooltips
    # constants
    legend_checkbox_width = 27  # px, width of the bokeh checkbox
    legend_glyph_width = legend_checkbox_width * 3  # px
    legend_disabled_color = "#E5E5E5"  # color of disabled legend items


class UI:
    path_separator = "."  # path separator for hist names
    # layout
    log_height = 50  # px, height of the log
    sidebar_width = 200  # px, width of the treeview and category selector
    # widgets
    multichoice_height = 40  # px, height of the multichoice widget
    numeric_input_width = 80  # px, width of the numeric input widget
    # colors
    background_color = "#E8E8E8"  # color of the background
    border_color = "#C8C8C8"  # color of the border
    border = f"1px solid {border_color}"  # border style
