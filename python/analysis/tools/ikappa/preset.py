from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Literal, Optional

# xrootd paths
# used for autocompletion
XRootD: list[str]

# # ratio: pairs
# # used for initialization
# # format: (name, (dataset1, dataset2))
# RatioPairs: ...  # TODO

# stack: groups
# used for initialization
# format: (name, [dataset1, dataset2, ...])
StackGroups: list[tuple[str, list[str]]]

# stack: allow duplicate processes
StackAllowDuplicate: bool

# model: patterns
# used for autocompletion and initialization
# format: {model: [pattern1, pattern2, ...]}
ModelPatterns: dict[str, list[str]]

# model: coupling sliders
# used for coupling sliders initialization
# format: {coupling: (min, max, step)}
# coupling: None for default
CouplingScan: dict[Optional[str], tuple[float, float, float]]

# palette
# format: hex color
Palette: list[str]

# palette: tint range
# format: (min, max)
# range: [0, 1]
PaletteTintRange: tuple[float, float]

# glyph: color alpha of fill
# range: [0, 1]
GlyphFillAlpha: float

# glyph: visibility
# used for initialization
# format: (dataset/hist, glyph)
# dataset/hist: regex or None for default
# glyph: "fill" or "step", "errorbar"
GlyphVisibility: list[tuple[Optional[str], Literal["fill", "step", "errorbar"]]]

# plotting: selected hists
# used for initialization
# format: regex pattern
SelectedHists: list[str]

# plotting: selected categories
# used for initialization
# format: [category1, category2, ...]
SelectedCategories: dict[str, list[str | int | bool]]


def update(module: ModuleType | dict):
    if isinstance(module, dict):
        mod = module
    else:
        mod = module.__dict__
    globals().update({k: v for k, v in mod.items() if k in __annotations__})


def reset():
    from . import _preset_default

    update(_preset_default)


reset()
