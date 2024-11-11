from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Literal

#
# xrootd paths
#
# used for autocompletion
XRootD: list[str]

#
# ratio: pairs
#
# used for initialization
# format: (name, (dataset1, dataset2))
RatioPairs: ...  # TODO

#
# stack: groups
#
# used for initialization
# format: (name, [dataset1, dataset2, ...])
StackGroups: list[tuple[str, list[str]]]

#
# model: patterns
#
# used for autocompletion and initialization
# format: {model: [pattern1, pattern2, ...]}
ModelPatterns: dict[str, list[str]]

# model: coupling sliders
#
# default min, max and step for coupling sliders
# used for initialization
# format: {coupling: (min, max, step)}
CouplingScan: dict[str, tuple[float, float, float]]

#
# plotting: palette
#
Palette: list[str]

#
# plotting: visible glyphs
#
# used for initialization
# format: (dataset, glyph)
VisibleGlyphs: list[tuple[str, Literal["fill", "step", "errorbar"]]]

#
# plotting: selected hists
#
# used for initialization
# format: regex pattern
SelectedHists: list[str]

#
# plotting: selected categories
#
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
