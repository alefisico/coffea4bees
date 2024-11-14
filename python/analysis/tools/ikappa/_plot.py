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

from . import preset
from ._bh import BHAxis, HistAxis
from ._hist import _FF, HistGroup
from ._utils import Component, Confirmation, DownloadLink, ExternalLink, PathInput
from ._widget import TreeView
from .config import UI

if TYPE_CHECKING:
    import numpy.typing as npt

    from .__main__ import Main
    from ._hist import Hist1D

_RESOURCE = ["cdn", "inline"]


class Profile(TypedDict):
    name: str
    rebin: int | list[int]


class _Profile:
    _KEYS = frozenset(Profile.__annotations__.keys()) - {"name"}

    def __init__(self, profile: Profile):
        self._keys = self._KEYS & profile.keys()
        self._profile = profile
        self._name = re.compile(profile["name"])

    def match(self, name: str) -> bool:
        return self._name.fullmatch(name) is not None

    def keys(self):
        return self._keys

    def __getitem__(self, key):
        return self._profile[key]


class Profiler:
    def __init__(self, profiles: list[Profile] = None):
        self._profiles = [*map(_Profile, profiles)] if profiles else []

    def generate(self, name: str) -> Profile:
        profile = {}
        for k in self._profiles:
            if k.match(name):
                profile.update(k)
        return profile


class AxisProjector(Component):
    def __init__(self, axis: HistAxis, dist: Hist, **kwargs):
        super().__init__(**kwargs)
        self._type = type(axis)
        self._name = axis.name
        self._label = axis.label or axis.name

        self._choices = BHAxis(flow=True, floating_format=_FF).labels(axis)
        self._indices: dict[str, int] = dict(
            zip(self._choices, range(len(self._choices)))
        )

        if selected := preset.SelectedCategories.get(self._name, []):
            selected = BHAxis(flow=True).indexof(axis, selected)
        if not selected:
            selected = [np.argmax(dist.values(flow=True))]

        self.dom = self.shared.nonempty(
            MultiChoice(
                title=self._label,
                value=[self._choices[i] for i in selected],
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
    flow: bool
    share: bool


class Plotter(Component):
    _DIV = {
        "font-size": "1.5em",
        "text-align": "center",
        "background-color": UI.background_color,
        "z-index": "-1",
        "border": UI.border,
    }
    _BUTTON = dict(
        sizing_mode="stretch_height",
        margin=(5, 0, 5, 5),
    )

    def __init__(self, parent: Main, **kwargs):
        super().__init__(**kwargs)
        self._data = None
        self._profile = Profiler()
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

        self._plot_queue: Queue[tuple[_PlotConfig, list[str]]] = Queue()
        self._plot_thread = Thread(target=self._plot, daemon=True)
        self._plot_thread.start()

    def reset(self):
        self.doc.add_next_tick_callback(self._dom_reset)

    def update_profile(self, profiles: list[Profile]):
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
        self._dom_plot = Button(label="Plot", button_type="success", **self._BUTTON)
        self._dom_plot.on_click(self._dom_select_plot)
        self._dom_profile = ExternalLink(
            shared=self.shared,
            label="Profile",
            button_type="primary",
            **self._BUTTON,
        )
        self._dom_profile.add_page(
            self._dom_check_profile,
            f"""
const columns = {str(["hist", *_Profile._KEYS])};
console.log(columns);
"""
            + """
const profile = JSON.parse(text);
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
text = `<!DOCTYPE html><html><head><title>Profile</title><style>
table, th, td {border: 1px solid black;border-collapse: collapse;}
th, td {padding: 5px;text-align: center;}
tr:hover {background-color: rgb(175, 225, 255);}
</style></head><body>` + table.outerHTML + "</body></html>";
""",
        )
        _SelectedHists = re.compile("|".join(map("({})".format, preset.SelectedHists)))
        selected_hists = [h for h in self.data if _SelectedHists.fullmatch(h)]
        self._dom_hist_select = self.shared.nonempty(
            self.shared.multichoice(
                value=selected_hists,
                options=[*self.data],
                search_option_limit=len(self.data),
            ),
        )
        # hist options
        self._dom_normalized = Checkbox(label="normalized", active=False)
        self._dom_density = Checkbox(
            label="density (Regular/Variable axis)", active=False
        )
        self._dom_log_y = Checkbox(label="log y-axis", active=False)
        self._dom_flow = Checkbox(label="under/overflow", active=False)
        self._dom_share = Checkbox(label="share canvas", active=False)
        # hist tree
        ncats = len(categories)
        self._dom_hist_tree = TreeView(
            selected=selected_hists,
            paths={k: f"hist{len(v.axes)-ncats}d" for k, v in self.data.items()},
            root="hists",
            separator=UI.path_separator,
            icons={
                "hist1d": "ti ti-chart-histogram",
                "hist2d": "ti ti-chart-scatter",
            },
            width=UI.sidebar_width,
            sizing_mode="stretch_height",
        )
        self._dom_hist_select.js_link("value", self._dom_hist_tree, "selected")
        self._dom_hist_tree.js_link("selected", self._dom_hist_select, "value")
        # select categories
        self._dom_cat_select = ScrollBox(
            child=column(
                width=UI.sidebar_width,
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
        self._dom_download = DownloadLink(
            shared=self.shared, **self.shared._icon_button_style
        )
        self._dom_download._icon.size = "1.5em"
        self._dom_download.add_page("plots", self._static_plot_page)
        self._dom_upload_preview = ExternalLink(
            shared=self.shared, **self.shared._icon_button_style
        )
        self._dom_upload_preview._icon.size = "1.5em"
        self._dom_upload_preview.add_page(self._static_plot_page)
        # main
        self.group.frozen = True
        self.main = row(
            self._dom_hist_tree,
            column(
                row(
                    self._dom_upload_path,
                    Div(text="Resource:", align="center"),
                    self._dom_upload_resource,
                    *self._dom_upload_confirm,
                    *self._dom_download,
                    *self._dom_upload_preview,
                    sizing_mode="stretch_width",
                ),
                row(
                    *self._dom_profile,
                    self._dom_plot,
                    self._dom_hist_select,
                    self._dom_full,
                    sizing_mode="stretch_width",
                ),
                row(
                    self._dom_normalized,
                    self._dom_density,
                    self._dom_log_y,
                    self._dom_flow,
                    self._dom_share,
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

        # render
        self.doc.add_next_tick_callback(self._dom_update)

    def status(self, msg: str):
        self.doc.add_next_tick_callback(partial(self._dom_update_status, msg))

    def _static_plot_page(self):
        if self._dom_content.child is self._dom_status:
            return ""
        return file_html(
            models=self._dom_content.child,
            resources=Resources(mode=self._dom_upload_resource.value),
            title="\u03BA Framework Plots",
        )

    def _dom_upload_plot(self):
        if self._dom_content.child is self._dom_status:
            self.log.error("No plot to upload.")
            return
        self._parent.upload(
            path=self._dom_upload_path.value,
            content=self._static_plot_page(),
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
        return json.dumps(
            {k: self._profile.generate(k) for k in self._dom_hist_select.value}
        )

    def _plot(self):
        while (task := self._plot_queue.get()) is not None:
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
        shared_axis = None
        bhaxis = BHAxis(flow=cfg["flow"], floating_format=_FF)
        for name in hists:
            hist = self.data[name]
            profile = self._profile.generate(name)
            try:
                self.status(f'Preparing histogram "{name}"...')
                match len(hist.axes) - len(self.categories):
                    case 1:
                        hist = self._1d_project(hist, cfg["flow"])
                        if "rebin" in profile:
                            hist = self._1d_rebin(hist, profile["rebin"], bhaxis)
                        if shared_axis is None:
                            shared_axis = hist[-1]
                        else:
                            if cfg["share"]:
                                if not bhaxis.equal(shared_axis, hist[-1]):
                                    raise RuntimeError(
                                        "Cannot plot histograms with different axes when sharing canvas."
                                    )
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

    def _1d_project(self, hist: Hist, flow: bool) -> Hist1D:
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
        _slice = slice(None)
        if not flow:
            under, over = BHAxis(flow=True, floating_format=_FF).flow(edge)
            _slice = slice(under, -1 if over else None)
        return (
            pd.DataFrame(val[_slice, :], columns=processes),
            pd.DataFrame(var[_slice, :], columns=processes),
            edge,
        )

    @staticmethod
    def __rebin(
        _idx: np.ndarray,
        *dfs: pd.DataFrame,
    ) -> Generator[pd.DataFrame, None, None]:
        for df in dfs:
            yield pd.DataFrame(
                np.add.reduceat(df.to_numpy(), _idx, axis=0), columns=df.columns
            )

    def _1d_rebin(self, hist: Hist1D, rebin: int | list[int], bhaxis: BHAxis) -> Hist1D:
        val, var, edge = hist
        idx = None
        try:
            edge, idx = bhaxis.rebin(edge, rebin)
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
            "flow": self._dom_flow.active,
            "share": self._dom_share.active,
        }
