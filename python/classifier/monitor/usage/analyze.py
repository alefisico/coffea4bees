import math
import re
from collections import defaultdict
from itertools import cycle, islice
from pathlib import Path
from typing import TypedDict

import fsspec
import networkx as nx
import numpy as np
import pandas as pd
from base_class.system.eos import EOS
from bokeh.embed import file_html
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    ColumnDataSource,
    CrosshairTool,
    CustomJS,
    CustomJSHover,
    CustomJSTickFormatter,
    HoverTool,
    Span,
    Toggle,
)
from bokeh.palettes import Set3_12 as PALETTE
from bokeh.plotting import figure
from bokeh.resources import CDN, INLINE
from pyvis.network import Network

from . import Checkpoint, ProcessInfo, Resource

_LABEL = {
    "cpu": "CPU",
    "memory": "Memory",
    "gpu": "GPU Memory",
}
_UNIT = {
    "cpu": "%",
    "memory": "MiB",
    "gpu": "MiB",
}

_VSPAN_WIDTH = 2
_VSPAN_COLOR = "red"
_GRAPH_KWARGS = dict(
    borderWidth=1,
    shape="box",
)

_TIMELINE_KWARGS = dict(
    x_axis_label="Time (s)",
    height=300,
    tools="xpan,xwheel_zoom,reset",
    sizing_mode="stretch_width",
)

_CODE = {
    "js": {
        "figure_visibility": "fig.visible = r.active && p.active;",
        "y_formatter": 'return special_vars.y.toFixed(2) + "{{ unit }}";',
        "open_url": 'window.open("{{ url }}", "{{ mode }}");',
    },
    "html": {},
}


def _file(file: str, ext: str = "js", **kwargs) -> str:
    if file not in _CODE[ext]:
        with open(Path(__file__).parent / ext / f"{file}.{ext}") as f:
            _CODE[ext][file] = f.read()
    content = _CODE[ext][file]
    if kwargs:
        kwargs = {"{{ " + k + " }}": v for k, v in kwargs.items()}
        content = re.sub(
            "|".join(re.escape(k) for k in kwargs),
            lambda match: kwargs.get(match.group(0)),
            content,
        )
    return content


def _js(file: str, **kwargs):
    return _file(file, "js", **kwargs)


def _html(file: str, **kwargs):
    return _file(file, "html", **kwargs)


class UsageData(TypedDict):
    checkpoints: list[Checkpoint]
    records: list[Resource]


class UsageDump(TypedDict):
    usage: dict[str, dict[int, UsageData]]
    process: dict[str, list[ProcessInfo]]


def _time_format(t: float) -> str:
    sign = "-" if t < 0 else ""
    t = abs(t)
    hour = int(t / 3600)
    minute = int(t % 3600 / 60)
    second = t % 60
    return f"{sign}{hour:02d}:{minute:02d}:{second:.3f}"


def _init():
    # initialize formatters
    time_tick = CustomJSTickFormatter(code=_js("time_formatter", timestamp="tick"))
    # initialize plot tools
    crosshair = CrosshairTool(
        dimensions="height",
        overlay=Span(dimension="height", line_dash="dashed", line_width=_VSPAN_WIDTH),
    )
    stack_tps: dict[str, HoverTool] = {}
    for unit in set(_UNIT.values()):
        stack_tp = HoverTool(
            tooltips=_html("stack_tooltip"),
            formatters={
                "@time": CustomJSHover(
                    code=_js("time_formatter", timestamp="special_vars.x")
                ),
                "$y": CustomJSHover(code=_js("y_formatter", unit=unit)),
                "$sy": CustomJSHover(
                    code=_js("stack_component_formatter", unit=unit)
                ),  # hacked
            },
        )
        stack_tp.renderers = []
        stack_tps[unit] = stack_tp
    checkpoints_tp = HoverTool(
        tooltips=_html("checkpoint_tooltip"),
        formatters={
            "@time": CustomJSHover(code=_js("time_formatter", timestamp="value")),
        },
    )
    checkpoints_tp.renderers = []
    # initialize toggles
    resource_btn = {"all": Toggle(label="All", active=True, button_type="primary")}
    for k in _LABEL:
        resource_btn[k] = Toggle(label=_LABEL[k], active=True, button_type="primary")
        resource_btn["all"].js_link("active", resource_btn[k], "active")
    return time_tick, crosshair, stack_tps, checkpoints_tp, resource_btn


def _plot_process_tree(ps: list[ProcessInfo], start_t: int):
    # construct dataframes
    df = pd.DataFrame(ps)
    df["start"] -= start_t / 1e9
    df["cmd"] = df["cmd"].apply("\n".join)
    df.sort_values("pid", inplace=True)
    df.drop_duplicates(subset=["pid", "parent"], inplace=True)
    df.loc[:, ["layer", "position"]] = 0
    df.reset_index(drop=True, inplace=True)
    known, unknown = set(), set()
    # construct graph
    G = nx.DiGraph()
    for _, p in df.iterrows():
        known.add(p["pid"])
        unknown.add(p["parent"])
        G.add_node(
            p["pid"],
            label=f'PID:{p["pid"]}\n{_time_format(p["start"])}',
            title=p["cmd"],
            group="known",
            **_GRAPH_KWARGS,
        )
        G.add_edge(p["parent"], p["pid"])
    for p in unknown - known:
        G.add_node(
            p,
            label=f"{p}",
            title=f"Unknown",
            group="unknown",
            **_GRAPH_KWARGS,
        )
    vis = Network(
        directed=True,
        select_menu=True,
        layout=True,
        cdn_resources="remote",
    )
    vis.from_nx(G)
    return vis.generate_html(notebook=False)


def _plot_stack(fig: figure, data: pd.DataFrame):
    return fig.varea_stack(
        sorted(set(data.columns) - {"time"}),
        x="time",
        source=ColumnDataSource(data=data),
        color=list(islice(cycle(PALETTE), len(data.columns) - 1)),
    )


def generate_report(dump: UsageDump, output: EOS, inline_resources: bool = False):
    resource = INLINE if inline_resources else CDN
    # data
    summary: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)
    checkpoints: dict[str, dict[int, pd.DataFrame]] = defaultdict(dict)
    records: dict[str, dict[int, dict[str, pd.DataFrame]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    start_t = math.inf
    # construct dataframes and find start time
    for ip, pids in dump["usage"].items():
        for pid, data in pids.items():
            c = pd.DataFrame(data["checkpoints"])
            c.sort_values("time", inplace=True)
            start_t = min(start_t, c["time"][0])
            checkpoints[ip][pid] = c
            r = data["records"]
            r = pd.DataFrame(r)
            r.sort_values("time", inplace=True)
            start_t = min(start_t, r["time"][0])
            for col in _LABEL:
                raw = pd.DataFrame(r[col].to_list())
                raw.fillna(0, inplace=True)
                records[ip][pid][col] = pd.concat([r["time"], raw], axis=1)
    for ip in sorted(records):
        pids = records[ip]
        # initialize utilities
        time_tick, crosshair, stack_tps, checkpoints_tp, resource_btns = _init()
        pid_btns = {"all": Toggle(label="All", active=True)}
        summary_btn = Button(label="Summary", button_type="success")
        summary_btn.js_on_event(
            "button_click",
            CustomJS(code=_js("open_url", url="../index.html", mode="_self")),
        )
        tree_btn = Button(label="Processes", button_type="warning")
        figs, links = [], {}
        # plot process tree
        if ip in dump["process"]:
            with fsspec.open(output / f"{ip}/process.html", mode="wt") as f:
                f.write(_plot_process_tree(dump["process"][ip], start_t))
            tree_btn.js_on_event(
                "button_click",
                CustomJS(code=_js("open_url", url=f"process.html", mode="_blank")),
            )
        # plot usage by pid
        for pid in sorted(pids):
            data = pids[pid]
            pid_btns[pid] = Toggle(label=f"PID:{pid}", active=True)
            pid_btns["all"].js_link("active", pid_btns[pid], "active")
            checkpoints[ip][pid]["time"] -= start_t
            for k in _LABEL:
                unit = _UNIT[k]
                # initialize figure
                fig: figure = figure(
                    title=f"{_LABEL[k]} PID:{pid}",
                    y_axis_label=f"{_LABEL[k]} ({unit})",
                    **_TIMELINE_KWARGS,
                    **links,
                )
                if not links:
                    links = {"x_range": fig.x_range}
                fig.add_tools(crosshair, stack_tps[unit], checkpoints_tp)
                fig.xaxis.formatter = time_tick

                # link to toggles
                figs.append(fig)
                activate = CustomJS(
                    args=dict(r=resource_btns[k], p=pid_btns[pid], fig=fig),
                    code=_js("figure_visibility"),
                )
                resource_btns[k].js_on_change("active", activate)
                pid_btns[pid].js_on_change("active", activate)

                # plot
                data[k]["time"] -= start_t
                stack_tps[unit].renderers.extend(_plot_stack(fig, data[k]))
                checkpoint = fig.vspan(
                    x="time",
                    color=_VSPAN_COLOR,
                    width=_VSPAN_WIDTH,
                    source=ColumnDataSource(data=checkpoints[ip][pid]),
                )
                checkpoints_tp.renderers.append(checkpoint)
        # align timestamps
        for k in _LABEL:
            timestamps = np.unique(
                np.concatenate([data[k]["time"] for data in pids.values()])
            )
            merged = pd.DataFrame(index=pd.Series(timestamps, name="time"))
            for pid, data in pids.items():
                data = data[k].copy(deep=False)
                data.set_index("time", inplace=True)
                for col in data.columns:
                    if col in merged:
                        merged.loc[data.index, col] = data[col]
                    else:
                        merged.loc[:, col] = np.interp(
                            timestamps, data.index, data[col], left=0, right=0
                        )
            merged.reset_index(inplace=True)
            summary[ip][k] = merged
        with fsspec.open(output / f"{ip}/usage.html", mode="wt") as f:
            f.write(
                file_html(
                    column(
                        [
                            row([*resource_btns.values(), tree_btn, summary_btn]),
                            row([*pid_btns.values()]),
                            *figs,
                        ],
                        sizing_mode="stretch_width",
                    ),
                    title=f"Usage {ip}",
                    resources=resource,
                )
            )
    # generate summary page
    time_tick, crosshair, stack_tps, _, resource_btns = _init()
    ip_btn = {"all": Toggle(label="All", active=True)}
    # plot usage by ip
    figs, links = [], {}
    for ip in sorted(summary):
        data = summary[ip]
        detail_btn = Button(label=f"IP:{ip}", button_type="success")
        detail_btn.js_on_event(
            "button_click",
            CustomJS(code=_js("open_url", url=f"{ip}/usage.html", mode="_self")),
        )
        ip_btn[ip] = Toggle(label=f"IP:{ip}", active=True)
        ip_btn["all"].js_link("active", ip_btn[ip], "active")
        ip_btn[ip].js_link("active", detail_btn, "visible")
        figs.append(detail_btn)
        for k in _LABEL:
            unit = _UNIT[k]
            # initialize figure
            fig: figure = figure(
                title=f"{_LABEL[k]} IP:{ip}",
                y_axis_label=f"{_LABEL[k]} ({unit})",
                **_TIMELINE_KWARGS,
                **links,
            )
            if not links:
                links = {"x_range": fig.x_range}
            fig.add_tools(crosshair, stack_tps[unit])
            fig.xaxis.formatter = time_tick

            # link to toggles
            figs.append(fig)
            activate = CustomJS(
                args=dict(r=resource_btns[k], p=ip_btn[ip], fig=fig),
                code=_js("figure_visibility"),
            )
            resource_btns[k].js_on_change("active", activate)
            ip_btn[ip].js_on_change("active", activate)

            # plot
            stack_tps[unit].renderers.extend(_plot_stack(fig, data[k]))
    with fsspec.open(output / "index.html", mode="wt") as f:
        f.write(
            file_html(
                column(
                    [
                        row([*resource_btns.values()]),
                        row([*ip_btn.values()]),
                        *figs,
                    ],
                    sizing_mode="stretch_width",
                ),
                title="Usage Summary",
                resources=resource,
            )
        )
