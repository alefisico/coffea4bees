from __future__ import annotations

import json
import re
from functools import partial
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING, Generator, Iterable, TypedDict

import numpy as np
import pandas as pd
from bokeh.embed import file_html
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    Checkbox,
    CustomJS,
    Div,
    MultiChoice,
    ScrollBox,
    Select,
)
from bokeh.resources import Resources
from hist import Hist
from hist.axis import AxesMixin, Regular, Variable

from ._hist import BHAxis, HistGroup
from ._treeview import TreeView
from ._utils import Component, Confirmation, PathInput
from .config import UI

if TYPE_CHECKING:
    import numpy.typing as npt

    from .__main__ import Main
    from ._hist import Hist1D

_RESOURCE = ["cdn", "inline"]


class Profile(TypedDict):
    rebin: int | list[int]


class Profiler:
    def __init__(self, profiles: dict[str, Profile]):
        self._profiles = [(re.compile(k), v) for k, v in profiles.items()]

    def generate(self, name: str) -> Profile:
        profile = {}
        for k, v in self._profiles:
            if k.fullmatch(name) is not None:
                profile.update(v)
        return profile


class AxisProjector(Component):
    def __init__(self, axis: AxesMixin, dist: Hist, **kwargs):
        super().__init__(**kwargs)
        self._type = type(axis)
        self._name = axis.name
        self._label = axis.label or axis.name

        self._choices = BHAxis.labels(axis)
        self._indices: dict[str, int] = dict(
            zip(self._choices, range(len(self._choices)))
        )

        self.dom = self.shared.nonempty(
            MultiChoice(
                title=self._label,
                value=[self._choices[np.argmax(dist.values(flow=True))]],
                options=self._choices,
                sizing_mode="stretch_width",
            )
        )

    def select(self, bins: Iterable[str]):
        return [self._indices[v] for v in bins]

    @property
    def selected(self):
        return [self._indices[v] for v in self.dom.value]


class _PlotConfig(TypedDict):
    normalized: bool
    density: bool
    log_y: bool


class Plotter(Component):
    _DIV = {
        "font-size": "1.5em",
        "text-align": "center",
        "background-color": UI.color_background,
        "z-index": "-1",
        "border": UI.border,
    }
    _BUTTON = dict(
        button_type="success",
        sizing_mode="stretch_height",
        margin=(5, 0, 5, 5),
    )

    def __init__(self, parent: Main, **kwargs):
        super().__init__(**kwargs)
        self._data = None
        self._profile = None
        self._parent = parent

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
        self.doc.add_next_tick_callback(self._dom_reset)

    def update_profile(self, profiles: dict[str, Profile]):
        self._profile = Profiler(profiles)

    def update_data(self, hists: dict[str, Hist], categories: set[str]):
        self.reset()
        self.data = hists

        # blocks
        self.categories: dict[str, AxisProjector] = {}
        test = next(iter(hists.values()))
        for axis in sorted(test.axes, key=lambda x: x.name):
            if axis.name in categories:
                self.categories[axis.name] = AxisProjector(
                    axis, test.project(axis.name), **self.inherit_global_states
                )
        self.group = HistGroup(
            self.categories, self._dom_enable_plot, **self.inherit_global_states
        )
        # select hists
        self._dom_full = self.shared.icon_button("arrows-maximize")
        self._dom_full.js_on_click(
            CustomJS(
                args=dict(
                    button=self._dom_full,
                    elements=[self.log.dom, self._parent._file_dom, self.group.dom],
                ),
                code="""
const full = button.icon.icon_name == "arrows-maximize";
button.icon.icon_name = full ? "arrows-minimize" : "arrows-maximize";
elements.forEach(e => e.visible = !full);
""",
            ),
        )
        self._dom_plot = Button(label="Plot", **self._BUTTON)
        self._dom_plot.on_click(self._dom_select_plot)
        self._dom_profile = Button(label="Profile", **self._BUTTON)
        self._dom_profile_div = Div(text="", visible=False)
        self._dom_profile.on_click(self._dom_check_profile)
        self._dom_profile_div.js_on_change(
            "text",
            CustomJS(
                args=dict(
                    div=self._dom_profile_div,
                    columns=["hist", *Profile.__annotations__.keys()],
                ),
                code="""
if (div.text != "") {
    const profile = JSON.parse(div.text);
    div.text = "";
    let table = document.createElement("table");
    let tr = table.insertRow();
    for (const col of columns) {
        let th = document.createElement("th");
        th.textContent = col;
        tr.appendChild(th);
    }
    for (const [k, row] of Object.entries(profile)) {
        tr = table.insertRow();
        for (const col of columns) {
            if (col == "hist") {
                let td = tr.insertCell();
                td.textContent = k;
            } else {
                let td = tr.insertCell();
                if (col in row) {
                    td.textContent = row[col];
                }
            }
        }
    }
    const blob = new Blob([
        `<!DOCTYPE html><html><head><title>Profile</title><style>
table, th, td {border: 1px solid black;border-collapse: collapse;}
th, td {padding: 5px;text-align: center;}
tr:hover {background-color: rgb(175, 225, 255);}
</style></head><body>` + table.outerHTML + "</body></html>"], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    window.open(url, "_blank");
    URL.revokeObjectURL(url);
}
""",
            ),
        )
        self._dom_hist_select = self.shared.nonempty(
            self.shared.multichoice(
                options=[*self.data], search_option_limit=len(self.data)
            )
        )
        # hist options
        self._dom_normalized = Checkbox(label="Normalized", active=False)
        self._dom_density = Checkbox(label="Density (Regular/Variable)", active=False)
        self._dom_log_y = Checkbox(label="Log y-axis", active=False)
        # hist tree
        ncats = len(categories)
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
        self._dom_content = ScrollBox(
            child=Div(),
            sizing_mode="stretch_both",
        )
        # upload
        self._dom_upload_path = PathInput(sizing_mode="stretch_width")
        self._dom_upload_resource = Select(value=_RESOURCE[0], options=_RESOURCE)
        self._dom_upload_confirm = Confirmation(self.shared, "upload")
        self._dom_upload_confirm.on_click(self._dom_upload_plot)
        # main
        self.group.frozen = True
        self.main = row(
            self._dom_hist_tree,
            column(
                row(
                    self._dom_upload_path,
                    Div(text="Resource:", align="center"),
                    self._dom_upload_resource,
                    self._dom_upload_confirm.dom,
                    sizing_mode="stretch_width",
                ),
                row(
                    self._dom_plot,
                    self._dom_profile,
                    self._dom_profile_div,
                    self._dom_hist_select,
                    self._dom_full,
                    sizing_mode="stretch_width",
                ),
                row(
                    self._dom_normalized,
                    self._dom_density,
                    self._dom_log_y,
                    sizing_mode="stretch_width",
                ),
                row(
                    self._dom_cat_select,
                    self._dom_content,
                    sizing_mode="stretch_both",
                ),
                sizing_mode="stretch_both",
            ),
            sizing_mode="stretch_both",
        )
        self.doc.add_next_tick_callback(self._dom_update)

        self._plot_queue: Queue[tuple[_PlotConfig, list[str]]] = Queue()
        self._plot_thread = Thread(target=self._plot, daemon=True)
        self._plot_thread.start()

    def status(self, msg: str):
        self.doc.add_next_tick_callback(partial(self._dom_update_status, msg))

    def _dom_upload_plot(self):
        if self._dom_content.child is self._dom_status:
            self.log.error("No plot to upload.")
            return
        self._parent.upload(
            path=self._dom_upload_path.value,
            content=file_html(
                models=self._dom_content.child,
                resources=Resources(mode=self._dom_upload_resource.value),
                title="\u03BA Framework Plots",
            ),
        )

    def _dom_update_status(self, msg: str):
        if self._dom_content.child is self._dom_status:
            self._dom_status.text = msg + "<br>" + self._dom_status.text
        else:
            self._dom_status.text = msg
            self._dom_content.child = self._dom_status

    def _dom_show_plot(self, plots):
        self._dom_content.child = plots

    def _dom_reset(self):
        self.dom.children = [self._dom_idle]

    def _dom_update(self):
        self._dom_update_status("")
        self.dom.children = [self.group.dom, self.main]

    def _dom_enable_plot(self, frozen: bool):
        self._dom_plot.disabled = not frozen
        self._dom_profile.disabled = not frozen
        self._dom_full.disabled = not frozen
        if frozen:
            self._dom_cat_select.child.children = [
                v.dom for k, v in self.categories.items() if k != self.group.process
            ]
        else:
            self._dom_cat_select.child.children = []
            self._dom_update_status("")

    def _dom_select_plot(self):
        self._plot_queue.put((self.config, self._dom_hist_select.value))

    def _dom_check_profile(self):
        if self._profile is not None:
            profile = {
                k: self._profile.generate(k) for k in self._dom_hist_select.value
            }
            self._dom_profile_div.text = json.dumps(profile)

    def _plot(self):
        while task := self._plot_queue.get():
            self._render(*task)

    def _render(self, cfg: _PlotConfig, hists: list[str]):
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
            profile = self._profile.generate(name) if self._profile else {}
            try:
                self.status(f'Preparing histogram "{name}"...')
                match len(hist.axes) - len(self.categories):
                    case 1:
                        hist = self._1d_project(hist)
                        if cfg["normalized"]:
                            hist = self._1d_normalized(hist)
                        if "rebin" in profile:
                            hist = self._1d_rebin(hist, profile["rebin"])
                        if cfg["density"]:
                            hist = self._1d_density(hist)
                        projected[name] = hist
                    case 2:  # TODO: plot 2D histogram
                        raise NotImplementedError("2D histogram is not supported yet.")
                    case _:
                        raise RuntimeError(
                            "Histogram with more than 2 dimensions is not supported."
                        )
            except Exception as e:
                self.log.error(f'Histogram "{name}" is skipped', exec_info=e)
        plots = self.group.render(data=projected, logger=self.status, **cfg)
        self.doc.add_next_tick_callback(partial(self._dom_show_plot, plots))

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

    def _1d_project(self, hist: Hist) -> Hist1D:
        val: npt.NDArray = hist.values(flow=True)
        var: npt.NDArray = hist.variances(flow=True)
        _transpose, edge = False, None
        _sum, _slice = [], []
        processes = [*self.group.selected]
        for i, axis in enumerate(hist.axes):
            n = axis.name
            if n == self.group.process:
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
            pd.DataFrame(val, columns=processes),
            pd.DataFrame(var, columns=processes),
            edge,
        )

    def _1d_normalized(self, hist: Hist1D) -> Hist1D:
        val, var, edge = hist
        total = val.sum(axis=0)
        val = val.div(total, axis=1)
        var = var.div(total**2, axis=1)
        return val, var, edge

    def _1d_density(self, hist: Hist1D) -> Hist1D:
        val, var, edge = hist
        if isinstance(edge, (Regular, Variable)):
            width = pd.Series(BHAxis.widths(edge))
            val = val.div(width, axis=0)
            var = var.div(width**2, axis=0)
        return val, var, edge

    @staticmethod
    def __rebin(
        _idx: np.ndarray,
        *dfs: pd.DataFrame,
    ) -> Generator[pd.DataFrame, None, None]:
        for df in dfs:
            yield pd.DataFrame(
                np.add.reduceat(df.to_numpy(), _idx, axis=0), columns=df.columns
            )

    def _1d_rebin(self, hist: Hist1D, rebin: int | list[int]) -> Hist1D:
        val, var, edge = hist
        idx = None
        try:
            edge, idx = BHAxis.rebin(edge, rebin)
        except ValueError as e:
            self.log.error(str(e))
        if idx is not None:
            val, var = self.__rebin(idx, val, var)
        return val, var, edge

    @property
    def config(self) -> _PlotConfig:
        return {
            "normalized": self._dom_normalized.active,
            "density": self._dom_density.active,
            "log_y": self._dom_log_y.active,
        }
