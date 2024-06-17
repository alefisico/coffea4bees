import operator as op
from functools import reduce
from itertools import chain
from typing import TypedDict

import pandas as pd
from base_class.utils import unique
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, HoverTool, Toggle
from bokeh.plotting import figure

from .code_cache import Importer

_NA = "N/A"
_ARROW = " ⇒ "

_SCATTER_SIZE = 10
_VSPAN_WIDTH = 3
_VSPAN_KWARGS = {
    "color": "green",
    "alpha": 0.3,
}
_FIGURE_KWARGS = {
    "height": 300,
    "sizing_mode": "stretch_width",
    "tools": "xpan,xwheel_zoom,reset,save",
}

code = Importer(__file__)


class StyleDict(TypedDict):
    line: dict[str, dict[str]]
    scatter: dict[str, dict[str]]


def generate_toggles(keys: set[str], category: dict[str, list[str]]):
    category, keys = category.copy(), keys.copy()
    category[None] = keys
    toggles: dict[str, Toggle] = dict()
    layout = []
    for k, v in category.items():
        group: list[Toggle] = []
        for vv in sorted(v):
            if vv in keys:
                keys.remove(vv)
                toggles[vv] = Toggle(label=vv, active=True, button_type="primary")
                group.append(toggles[vv])
        if group:
            if k is None:
                k = "other"
            group.insert(0, Toggle(label=k, active=True, button_type="success"))
            for toggle in group[1:]:
                group[0].js_link("active", toggle, "active")
            layout.extend(group)
    return toggles, row(*layout)


def plot_multiphase_scalar(
    *,
    plot: list[str],
    phase: pd.DataFrame,
    data: dict[tuple[str, ...], pd.DataFrame],
    style: StyleDict,
    category: dict[str, list[str]] = None,
    shared_category_toggles: dict[str, Toggle] = None,
):
    layout = []
    # generate toggles
    if shared_category_toggles is not None:
        cat_toggles = shared_category_toggles
    elif category is not None:
        cat_toggles, toggle_row = generate_toggles(
            set(chain.from_iterable(data.keys())), category
        )
        layout.append(toggle_row)
    else:
        raise ValueError("Either category or shared_category_toggles must be provided.")
    # generate phase separator
    separators = {"x": [], "label": []}
    _nulls = phase.isnull()
    _phase = phase.astype(str, copy=True)
    # https://github.com/pandas-dev/pandas/issues/20442
    _phase[_nulls] = _NA
    _shifted = _phase.shift(1, fill_value=_NA)
    _changed = _phase != _shifted
    _selection = _changed.any(axis=1)
    _phase, _shifted = _phase[_selection], _shifted[_selection]
    _labels = pd.DataFrame(index=_phase.index)
    for p in _phase.columns:
        _labels[p] = _shifted[p].str.cat(_phase[p], sep=_ARROW)
    _labels[~_changed] = None
    separators = {"x": [], "label": [], "width": []}
    for i, r in _labels.iterrows():
        r = r.dropna()
        separators["x"].append(i - 0.5)
        separators["label"].append(
            "\n".join(code.html("custom_tooltip", key=k, value=v) for k, v in r.items())
        )
        separators["width"].append(len(r) * _VSPAN_WIDTH)
    del _phase, _nulls, _shifted, _changed, _labels, _selection
    # plot data
    dfs = {k: pd.concat([v, phase], axis=1) for k, v in data.items()}
    columns = unique(chain.from_iterable(map(lambda x: x.columns, dfs.values())))
    scatter_hover = HoverTool(
        tooltips=[(k, "@{" + k + "}") for k in columns],
    )
    scatter_hover.renderers = []
    phase_hover = HoverTool(tooltips="@label")
    phase_hover.renderers = []
    shared_x_range = None
    for p in plot:
        fig = figure(
            title=p,
            y_axis_label=p,
            **_FIGURE_KWARGS,
        )
        fig.add_tools(scatter_hover, phase_hover)
        if shared_x_range is None:
            shared_x_range = fig.x_range
        else:
            fig.x_range = shared_x_range
        for k, df in dfs.items():
            l = fig.line(
                source=ColumnDataSource(data=df),
                x="index",
                y=p,
                legend_label=",".join(k),
                **reduce(op.or_, (style["line"][k].get(s, {}) for s in k)),
            )
            s = fig.scatter(
                source=ColumnDataSource(data=df),
                x="index",
                y=p,
                **reduce(op.or_, (style["scatter"][k].get(s, {}) for s in k)),
                size=_SCATTER_SIZE,
            )
            togs = [cat_toggles[kk] for kk in k]
            activate = CustomJS(
                args=dict(toggles=togs, curves=[l, s]), code=code.js("curve_visibility")
            )
            for tog in togs:
                tog.js_on_change("active", activate)
            scatter_hover.renderers.append(s)
        vs = fig.vspan(
            x="x",
            width="width",
            source=ColumnDataSource(data=separators),
            **_VSPAN_KWARGS,
        )
        phase_hover.renderers.append(vs)
        layout.append(fig)

    return column(
        *layout,
        sizing_mode="stretch_width",
    )


def plot_multiphase_curve(): ...
