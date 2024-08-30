from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Generator, Iterable

import numpy as np
import pandas as pd
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

if TYPE_CHECKING:
    import numpy.typing as npt

    from ._hist import Hist1D


# Regular, Variable
class AxisProjector:
    _TRUE = "True"
    _FALSE = "False"
    _OTHERS = '<span style="font-style: italic; font-weight: bold;">Others</span>'

    @classmethod
    def __bins(cls, axis: AxesMixin):
        _ax = axis._ax  # Note: private attribute in boost-histogram
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

    def __init__(self, axis: AxesMixin, dist: Hist):
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
            value=[self._choices[np.argmax(dist.values(flow=True))]],
            options=self._choices,
            sizing_mode="stretch_width",
        )
        self.dom.on_change("value", self._dom_change)

    def _dom_change(self, attr, old, new):
        if not new:
            self.dom.stylesheets = [self._style_empty]
        else:
            self.dom.stylesheets = []

    def select(self, bins: Iterable[str]):
        return [self._indices[v] for v in bins]

    @property
    def selected(self):
        return [self._indices[v] for v in self.dom.value]


class Plotter:
    _DIV = {
        "font-size": "1.5em",
        "text-align": "center",
        "background-color": UI.color_background,
        "z-index": "-1",
    }

    def __init__(self, doc: Document, logger: BokehLog, parent):
        self._doc = doc
        self._data = None
        self._parent = parent

        self.log = logger

        self._dom_idle = Div(
            text="Waiting for data...",
            sizing_mode="stretch_both",
            styles=self._DIV,
        )
        self._dom_status = Div(
            text="",
            sizing_mode="stretch_both",
            styles=self._DIV,
        )
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
            height=UI.height_select_bar,
            search_option_limit=len(self.data),
            sizing_mode="stretch_width",
        )
        ncats = len(categories)
        # select hists
        self._dom_hist_tree = TreeView(
            paths={k: f"hist{len(v.axes)-ncats}d" for k, v in self.data.items()},
            root="hists",
            separator=".",
            icons={
                "hist1d": "bi-bar-chart-line",
                "hist2d": "bi-boxes",
            },
            width=UI.width_side,
            sizing_mode="stretch_height",
        )
        self._dom_hist_select.js_link("value", self._dom_hist_tree, "selected")
        self._dom_hist_tree.js_link("selected", self._dom_hist_select, "value")
        # select categories
        self._dom_cat_select = ScrollBox(
            child=column(
                width=UI.width_side,
                sizing_mode="stretch_height",
            ),
            sizing_mode="stretch_height",
        )
        # plots
        self._dom_main = ScrollBox(
            child=Div(),
            sizing_mode="stretch_both",
        )
        # blocks
        self.categories: dict[str, AxisProjector] = {}
        test = next(iter(hists.values()))
        for axis in sorted(test.axes, key=lambda x: x.name):
            if axis.name in categories:
                self.categories[axis.name] = AxisProjector(
                    axis, test.project(axis.name)
                )
        self.hist = HistGroup(self.categories, self.log, self._dom_enable_plot)
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
                    self._dom_main,
                    sizing_mode="stretch_both",
                ),
                sizing_mode="stretch_both",
            ),
            sizing_mode="stretch_both",
        )

        self._doc.add_next_tick_callback(self._dom_update)

    def status(self, msg: str):
        self._doc.add_next_tick_callback(partial(self._dom_update_status, msg))

    def _dom_update_status(self, msg: str):
        self._dom_main.child = self._dom_status
        self._dom_status.text = msg

    def _dom_show_plot(self, plots):
        self._dom_main.child = plots

    def _dom_reset(self):
        self.dom.children = [self._dom_idle]

    def _dom_update(self):
        self._dom_update_status("")
        self.full = False

    def _dom_enable_plot(self, frozen: bool):
        self._dom_plot.disabled = not frozen
        self._dom_full.disabled = not frozen
        if frozen:
            self._dom_cat_select.child.children = [
                v.dom for k, v in self.categories.items() if k != self.hist.process
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
        self.status("Plotting...")
        errs = []
        for k, v in self.categories.items():
            if not v.selected:
                errs.append(f'"{k}" is empty.')
        if not hists:
            errs.append("No histogram selected.")
        if errs:
            return self.log.error(*errs)

        projected = {}
        for name in hists:
            hist = self.data[name]
            try:
                match len(hist.axes) - len(self.categories):
                    case 1:
                        projected[name] = self._project1d(hist)
                    case 2:  # TODO: plot 2D histogram
                        raise NotImplementedError("2D histogram is not supported yet.")
                    case _:
                        raise RuntimeError(
                            "Histogram with more than 2 dimensions is not supported."
                        )
            except Exception as e:
                self.log.error(f'Histogram "{name}" is skipped', exec_info=e)
        plots = self.hist(projected, self.status)
        # TODO save or show
        # TEMP below
        self._doc.add_next_tick_callback(partial(self._dom_show_plot, plots))

    @staticmethod
    def __project(
        _slice: Iterable[npt.NDArray],
        _sum: Iterable[int],
        _transpose: bool,
        *arrs: npt.NDArray,
    ) -> Generator[npt.NDArray, None, None]:
        _slice = np.ix_(*_slice)
        _sum = tuple(_sum)
        for arr in arrs:
            arr = np.sum(arr[_slice], axis=_sum)
            if _transpose:
                arr = arr.T
            yield arr

    def _project1d(self, hist: Hist) -> Hist1D:
        val: npt.NDArray = hist.values(flow=True)
        var: npt.NDArray = hist.variances(flow=True)
        _transpose, edge = False, None
        _sum, _slice = [], []
        processes = [*self.hist.selected]
        for i, axis in enumerate(hist.axes):
            n = axis.name
            if n == self.hist.process:
                _slice.append(self.categories[n].select(processes))
                _transpose = edge is None
            elif n in self.categories:
                _slice.append(self.categories[n].selected)
                _sum.append(i)
            else:
                _slice.append(np.arange(val.shape[i]))
                edge = axis
        val, var = self.__project(_slice, _sum, _transpose, val, var)
        return (
            pd.DataFrame(var, columns=processes),
            pd.DataFrame(val, columns=processes),
            edge,
        )
