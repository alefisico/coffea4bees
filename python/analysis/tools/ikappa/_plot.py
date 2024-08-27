from typing import Iterable

import numpy as np
import numpy.typing as npt
from bokeh.document import Document
from bokeh.layouts import column, row
from bokeh.models import Button, Div, InlineStyleSheet, MultiChoice, ScrollBox
from hist import Hist
from hist.axis import (
    AxesMixin,
    Boolean,
    IntCategory,
    Integer,
    Regular,
    StrCategory,
    Variable,
)

from ._hist import HistGroup
from ._treeview import TreeView
from ._utils import BokehLog
from .config import UI


# Regular, Variable
class AxisProjector:
    _TRUE = "True"
    _FALSE = "False"
    _OTHERS = '<span style="font-style: italic; font-weight: bold;">Others</span>'

    @classmethod
    def __bins(cls, axis: AxesMixin):
        _ax = axis._ax
        over = _ax.traits_overflow
        under = _ax.traits_underflow
        cats = [*axis]
        match axis:
            case Boolean():
                return [cls._FALSE, cls._TRUE]
            case IntCategory():
                return [*map(str, cats)] + ([cls._OTHERS] if over else [])
            case StrCategory():
                return cats + ([cls._OTHERS] if over else [])
            case Integer():
                return (
                    ([f"<{cats[0]}"] if under else [])
                    + [*map(str, cats)]
                    + ([f">{cats[-1]}"] if over else [])
                )
            case Regular() | Variable():
                _cats = []
                for a, b in cats[:-1]:
                    _cats.append(f"[{a},{b})")
                a, b = cats[-1]
                _cats.append(f"[{a},{b}{')' if over else ']'}")
                return (
                    ([f"(-\u221E,{cats[0][0]})"] if under else [])
                    + _cats
                    + ([f"[{cats[-1][-1]}, \u221E)"] if over else [])
                )

    def __init__(self, axis: AxesMixin):
        self._type = type(axis)
        self._name = axis.name
        self._label = axis.label or axis.name

        self._choices = self.__bins(axis)
        self._indices: dict[str, int] = dict(
            zip(self._choices, range(len(self._choices)))
        )

        self._style_empty = InlineStyleSheet(
            css="div.choices__inner {background-color: pink;}"
        )
        self.dom = MultiChoice(
            title=self._label,
            value=self._choices[-1:],
            options=self._choices,
            sizing_mode="stretch_width",
        )
        self.dom.on_change("value", self._dom_change)

    def _dom_change(self, attr, old, new):
        if not new:
            self.dom.stylesheets = [self._style_empty]
        else:
            self.dom.stylesheets = []

    def select(self, *bins: str):
        return [self._indices[v] for v in bins]

    @property
    def selected(self):
        return [self._indices[v] for v in self.dom.value]


class _Projector:
    def __init__(self, axes: Iterable[AxesMixin], categories: set[str]):
        self.axes: dict[str, AxisProjector] = {
            axis.name: AxisProjector(axis)
            for axis in sorted(axes, key=lambda x: x.name)
            if axis.name in categories
        }

    def __call__(
        self, hist: Hist, exclude: set[str] = None
    ) -> tuple[npt.NDArray, npt.NDArray, list[AxesMixin]]:
        exclude = set(exclude or ())
        val = hist.values(flow=True)
        var = hist.variances(flow=True)
        edges, _sum, _slice = [], [], []
        for i, axis in enumerate(hist.axes):
            n = axis.name
            if n in self.axes and n not in exclude:
                _slice.append(self.axes[n].selected)
                _sum.append(i)
            else:
                _slice.append(np.arange(val.shape[i]))
                edges.append(axis)
        sums = tuple(_sum)
        slicer = np.ix_(*_slice)
        return (
            np.sum(val[slicer], axis=sums),
            np.sum(var[slicer], axis=sums),
            edges,
        )


class Plotter:
    def __init__(self, doc: Document, logger: BokehLog, parent):
        self._doc = doc
        self._data = None
        self._parent = parent

        self.log = logger
        self.dom = column(sizing_mode="stretch_both")
        self._dom_reset()

    def reset(self):
        self._doc.add_next_tick_callback(self._dom_reset)

    def update(self, hists: dict[str, Hist], categories: set[str]):
        self.reset()
        self.data = hists

        self._dom_full = self.log.ibtn("")
        self._dom_full.on_click(self._dom_fullscreen)
        self._dom_plot = Button(
            label="Plot", button_type="success", sizing_mode="stretch_height"
        )
        self._dom_plot.on_click(self._dom_plot_selected)
        self._dom_hist_select = MultiChoice(
            options=[*self.data],
            search_option_limit=len(self.data),
            sizing_mode="stretch_width",
        )
        ncats = len(categories)
        self._dom_hist_tree = TreeView(
            paths={k: f"hist{len(v.axes)-ncats}d" for k, v in self.data.items()},
            root="hists",
            separator=".",
            icons={
                "hist1d": "bi-bar-chart-line",
                "hist2d": "bi-boxes",
            },
            width=UI.side_width,
            sizing_mode="stretch_height",
        )
        self._dom_hist_select.js_link("value", self._dom_hist_tree, "selected")
        self._dom_hist_tree.js_link("selected", self._dom_hist_select, "value")
        # select category
        self._dom_cat_select = ScrollBox(
            child=column(
                width=UI.side_width,
                sizing_mode="stretch_height",
            ),
            sizing_mode="stretch_height",
        )
        # blocks
        self.project = _Projector(next(iter(hists.values())).axes, categories)
        self.hist = HistGroup(self.project.axes, self.log, self._dom_enable_plot)
        self.hist.frozen = True
        self._main_dom = row(
            self._dom_hist_tree,
            column(
                row(
                    self._dom_plot,
                    self._dom_hist_select,
                    self._dom_full,
                    sizing_mode="stretch_width",
                ),
                row(
                    self._dom_cat_select,
                    sizing_mode="stretch_both",
                ),
                sizing_mode="stretch_both",
            ),
            sizing_mode="stretch_both",
        )

        self._doc.add_next_tick_callback(self._dom_update)

    def _dom_reset(self):
        self.dom.children = [Div(text="Waiting for data...")]

    def _dom_update(self):
        self.full = False

    def _dom_enable_plot(self, frozen: bool):
        self._dom_plot.disabled = not frozen
        self._dom_full.disabled = not frozen
        if frozen:
            self._dom_cat_select.child.children = [
                v.dom for k, v in self.project.axes.items() if k != self.hist.process
            ]
        else:
            self._dom_cat_select.child.children = []

    def _dom_fullscreen(self):
        self.full = not self.full
        self._parent.full = self.full

    def _dom_plot_selected(self):
        self._plot(self._dom_hist_select.value)

    @property
    def full(self):
        return self._full

    @full.setter
    def full(self, value):
        self._full = value
        if value:
            self._dom_full.icon.icon_name = "arrows-minimize"
            self.dom.children = [self._main_dom]
        else:
            self._dom_full.icon.icon_name = "arrows-maximize"
            self.dom.children = [self.hist.dom, self._main_dom]

    def _plot(self, hists: list[str]):
        # TODO: plot or save
        empty = False
        for k, v in self.project.axes.items():
            if not v.selected:
                self.log.error(f'"{k}" is empty.')
                empty = True
        if not hists:
            self.log.error("No histogram selected.")
            empty = True
        if empty:
            return

        self.hist(
            {k: self.project(self.data[k], exclude=(self.hist.process,)) for k in hists}
        )
