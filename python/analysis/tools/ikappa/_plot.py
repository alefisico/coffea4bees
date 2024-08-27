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

from ._legend import LegendGroup
from ._treeview import TreeView
from ._utils import BokehLog
from .config import UI


# Regular, Variable
class AxisProjector:
    _TRUE = "True"
    _FALSE = "False"
    _OTHERS = "Others"

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
                if not over:
                    return cats
                _cats = set(cats)
                others = cls._OTHERS
                while others in _cats:
                    others += "?"
                return cats + [others]
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

    @property
    def selected(self):
        return [self._indices[v] for v in self.dom.value]


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
        test = next(iter(hists.values()))

        self.hists = hists
        self.categories: dict[str, AxisProjector] = {
            axis.name: AxisProjector(axis)
            for axis in sorted(test.axes, key=lambda x: x.name)
            if axis.name in categories
        }
        n_cat = len(self.categories)

        self._dom_full = self.log.ibtn("")
        self._dom_full.on_click(self._dom_fullscreen)
        self._dom_plot = Button(
            label="Plot", button_type="success", sizing_mode="stretch_height"
        )
        self._dom_plot.on_click(self._dom_plot_selected)
        self._dom_hist_select = MultiChoice(
            options=[*self.hists],
            search_option_limit=len(self.hists),
            sizing_mode="stretch_width",
        )
        self._dom_hist_tree = TreeView(
            paths={k: f"hist{len(v.axes)-n_cat}d" for k, v in self.hists.items()},
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
        self.groups = LegendGroup(self.categories, self.log, self._dom_enable_plot)
        self.groups.frozen = True
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
                v.dom for k, v in self.categories.items() if k != self.groups.process
            ]
        else:
            self._dom_cat_select.child.children = []

    def _dom_fullscreen(self):
        self.full = not self.full
        self._parent.full = self.full

    def _dom_plot_selected(self):
        self._plot()  # TODO: implement this method

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
            self.dom.children = [self.groups.dom, self._main_dom]

    def _plot(self): ...  # TODO: implement this method
