from types import MappingProxyType
from typing import Literal

#
# xrootd paths
#
# used for autocompletion
XRootD: tuple[str, ...] = (
    "root://cmseos.fnal.gov//store/",
    "root://eosuser.cern.ch//eos/",
)

#
# stack: groups
#
# used for initialization
# format: (name, (dataset1, dataset2, ...))
StackGroups: tuple[tuple[str, tuple[str, ...]]] = (
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

#
# model: patterns
#
# used for autocompletion and initialization
# format: {model: (pattern1, pattern2, ...)}
ModelPatterns: dict[str, tuple[str, ...]] = MappingProxyType(
    {
        "ggF": (r"^GluGluToHHTo4B_cHHH(?P<kl>[p\d]+)$",),
    }
)

#
# model: coupling sliders
#
# default min, max and step for coupling sliders
# used for initialization
# format: {coupling: (min, max, step)}
CouplingScan: dict[str, tuple[float, float, float]] = MappingProxyType(
    {
        "kl": (-5, 10, 0.1),
        None: (0, 5, 0.1),  # all other couplings
    }
)

#
# plotting: palette
#
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

#
# plotting: visible glyphs
#
# used for initialization
# format: (dataset, glyph)
VisibleGlyphs: tuple[tuple[str, Literal["fill", "step", "errorbar"]]] = (
    ("data", "errorbar"),
    ("QCD Multijet", "fill"),
    ("TTTo2L2Nu", "fill"),
    ("TTToSemiLeptonic", "fill"),
    ("TTToHadronic", "fill"),
    ("ZH4b", "step"),
    ("ZZ4b", "step"),
)

#
# plotting: selected hists
#
# used for initialization
# format: regex pattern
SelectedHists: tuple[str, ...] = (r"SvB_MA\.ps.*fine",)
