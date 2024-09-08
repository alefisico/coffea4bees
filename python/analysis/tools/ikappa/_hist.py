from __future__ import annotations

import difflib
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import chain, cycle
from typing import TYPE_CHECKING, Callable, Generator, Iterable, NamedTuple, Optional

import numpy as np
from base_class.physics import di_higgs
from base_class.physics.di_higgs import Coupling, Diagram
from bokeh.layouts import column, row
from bokeh.models import (
    AutocompleteInput,
    Button,
    Checkbox,
    ColumnDataSource,
    CustomJS,
    Div,
    Dropdown,
    HoverTool,
    MultiChoice,
    Row,
    ScrollBox,
    Select,
    Slider,
    TextInput,
    Whisker,
)
from bokeh.plotting import figure
from hist.axis import (
    AxesMixin,
    Boolean,
    IntCategory,
    Integer,
    Regular,
    StrCategory,
    Variable,
)

from ._utils import RGB, Component
from .config import UI, CouplingScan, Datasets, Palette, Stacks
from .config import FloatFormat as _FF

if TYPE_CHECKING:
    import pandas as pd

    from ._plot import AxisProjector

    class Hist1D(NamedTuple):
        values: pd.DataFrame
        variances: pd.DataFrame
        edge: AxesMixin

    class Hist2D(NamedTuple):
        values: pd.DataFrame
        variances: pd.DataFrame
        edges: tuple[AxesMixin, AxesMixin]


_DIHIGGS = sorted(set(di_higgs.__all__) - {"Coupling"})
_PROCESS = "process"
_WHITE = RGB(255, 255, 255)
_BLACK = RGB(0, 0, 0)

_CODES = {
    "visibility": """
let visible = true;
for (let check of checks) {
    visible = visible && check.active;
}
for (let glyph of glyphs) {
    glyph.visible = visible;
}
"""
}

_HIST_FILL = dict(
    left="left",
    right="right",
    line_width=0,
)

_HIST_STEP_LEFT = dict(
    x="left",
    mode="after",
    line_width=1,
)

_HIST_STEP_RIGHT = dict(
    x="right",
    mode="before",
    line_width=1,
)

_HIST_ERRORBAR = dict(
    base="center",
    level="annotation",
    line_width=1,
)

_BOX_STYLE = {
    "border": UI.border,
    "margin-top": "-1px",
}

_badge = '<span class="badge" style="background-color: {color};">{text}</span>'.format


class SourceID:
    regular = "rp{:d}".format
    model = "m{}p{:d}".format
    basis = "m{}p{:d}b{:d}".format
    stack = "s{:d}p{:d}".format


class BHAxis:
    _TRUE = "True"
    _FALSE = "False"
    _OTHERS = '<span style="font-style: italic; font-weight: bold;">Others</span>'

    @classmethod
    def flow(cls, axis: AxesMixin) -> tuple[bool, bool]:
        _ax = axis._ax  # Note: Accessing private attribute in boost-histogram
        return _ax.traits_underflow, _ax.traits_overflow

    @classmethod
    def widths(cls, axis: Regular | Variable) -> list[float]:
        under, over = cls.flow(axis)
        width = [*axis.widths]
        if under:
            width.insert(0, np.inf)
        if over:
            width.append(np.inf)
        return width

    @classmethod
    def edges(cls, axis: Regular | Variable, finite: bool = False) -> list[float]:
        under, over = cls.flow(axis)
        edges = [*axis.edges]
        flow = np.min(axis.widths) if finite else np.inf
        if under:
            edges.insert(0, edges[0] - flow)
        if over:
            edges.append(edges[-1] + flow)
        return edges

    @classmethod
    def labels(cls, axis: AxesMixin) -> list[str]:
        under, over = BHAxis.flow(axis)
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
                    _cats.append(f"[{_FF(a)},{_FF(a)})")
                a, b = cats[-1]
                _cats.append(f"[{_FF(a)},{_FF(b)}{')' if over else ']'}")
                return (
                    ([f"(-\u221E,{_FF(cats[0][0])})"] if under else [])
                    + _cats
                    + ([f"[{_FF(cats[-1][-1])}, \u221E)"] if over else [])
                )


def _label(text: str):
    return Div(text=text, align="center")


def _btn(label: str, *onclick):
    btn = Button(label=label, button_type="primary", align="center")
    for click in onclick:
        if isinstance(click, CustomJS):
            btn.js_on_click(click)
        else:
            btn.on_click(click)
    return btn


def _palette(n: int):
    selected = cycle(Palette)
    return [next(selected) for _ in range(n)]


class _KappaMatch:
    _D = re.compile(r"\D")

    def __init__(
        self, pattern: str = "", processes: Iterable[str] = (), model: str = ""
    ):
        self._match: dict[str, dict[str, float]] = {}
        try:
            for process in processes:
                match = re.match(pattern, process)
                if match is not None:
                    self._match[process] = dict(map(self._f, match.groupdict().items()))
            self._couplings = Coupling(*self._match.values())
            self._model: Diagram = getattr(di_higgs, model)(
                basis=self._couplings, unit_basis_weight=True
            )
        except Exception:
            self._match = {}

    def __bool__(self):
        return len(self._match) > 0

    @classmethod
    def _f(cls, kv: tuple[str, str]):
        k, v = kv
        return k, float(cls._D.sub(".", v))


class _Matched(ABC):
    dom_hint: list
    dom_hide: list
    dom_matched: Dropdown
    dom: Row
    matched: bool

    @abstractmethod
    def __iter__(self): ...

    def _disable(self, value: bool):
        if value:
            matched = sorted(self) if self.matched else []
            self.dom_matched.menu = [(m, m) for m in matched]
            self.dom_matched.label = "Matched" if matched else "No Match"
            self.dom_matched.button_type = "success" if matched else "danger"

            self.dom.children = self.dom_hint + [self.dom_matched]
        else:
            self.dom.children = self.dom_hint + self.dom_hide


class _KappaModel(_Matched):
    badge = _badge(color="#385CB4", text="model")
    matched: _KappaMatch

    def __init__(self, model: str, pattern: str, matched: _KappaMatch = None):
        self.matched = _KappaMatch() if matched is None else matched

        self._dom_model = Select(options=_DIHIGGS, align="center")
        self._dom_patterns = AutocompleteInput(
            value=pattern,
            restrict=False,
            sizing_mode="stretch_width",
            align="center",
            min_characters=0,
        )
        self._dom_model.js_on_change(
            "value",
            CustomJS(
                args=dict(
                    patterns={k: list(v) for k, v in Datasets.items()},
                    select=self._dom_model,
                    input=self._dom_patterns,
                ),
                code="input.completions = patterns[select.value] || [];",
            ),
        )
        self._dom_model.value = model

        self.dom_hint = [_label(text="Model:"), self._dom_model]
        self.dom_hide = [_label(text="Pattern:"), self._dom_patterns]
        self.dom_matched = Dropdown(sizing_mode="stretch_width", align="center")
        self.dom = row(sizing_mode="stretch_width")

        self.disabled = False

    def __iter__(self):
        yield from self.matched._match

    def __contains__(self, item):
        return item in self.matched._match

    def update(self, processes: Iterable[str]):
        self.matched = _KappaMatch(
            self._dom_patterns.value, processes, self._dom_model.value
        )

    @property
    def disabled(self):
        return self._dom_model.disabled

    @disabled.setter
    def disabled(self, value: bool):
        self._dom_model.disabled = value
        self._dom_patterns.disabled = value
        self._disable(value)

    @property
    def name(self):
        return self._dom_patterns.value

    @property
    def model(self):
        return self._dom_model.value


class _StackGroup(_Matched):
    badge = _badge(color="#29855A", text="stack")
    matched: list[str]

    def __init__(
        self, name: str, processes: MultiChoice, matched: Iterable[str] = None
    ):
        self.matched = list(matched or ())

        self._dom_name = TextInput(value=name or "")
        self._dom_bins = MultiChoice(
            value=self.matched,
            options=processes.value,
            sizing_mode="stretch_width",
            align="center",
        )
        self._dom_bins.js_link("options", processes, "value")

        self.dom_hint = [_label(text="Stack:"), self._dom_name]
        self.dom_hide = [_label(text="Bins:"), self._dom_bins]
        self.dom_matched = Dropdown(sizing_mode="stretch_width", align="center")
        self.dom = row(sizing_mode="stretch_width")

        self.disabled = False

    def __iter__(self) -> Generator[str, None, None]:
        yield from self.matched

    def __contains__(self, item):
        return item in self.matched

    def update(self, processes: Iterable[str]):
        if set(self._dom_bins.value) <= set(processes):
            self.matched = self._dom_bins.value
        else:
            self.matched = []

    @property
    def disabled(self):
        return self._dom_name.disabled

    @disabled.setter
    def disabled(self, value: bool):
        self._dom_name.disabled = value
        self._dom_bins.disabled = value
        self._disable(value)

    @property
    def name(self):
        return self._dom_name.value


class HistGroup(Component):
    _FREEZE = {
        False: dict(label="Setup", button_type="success"),
        True: dict(label="Unset", button_type="danger"),
    }

    def __init__(
        self,
        categories: dict[str, AxisProjector],
        *freeze_callbacks: Callable[[bool], None],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._callbacks = freeze_callbacks
        self._categories = dict(
            filter(lambda x: x[1]._type is StrCategory, categories.items())
        )
        # controls
        self._dom_freeze = _btn("", self._dom_freeze_click)
        self._dom_add_model = self.shared.icon_button("plus", self._dom_add_model_click)
        self._dom_remove_model = self.shared.icon_button(
            "minus", self._dom_remove_model_click
        )
        self._dom_add_stack = self.shared.icon_button("plus", self._dom_add_stack_click)
        self._dom_remove_stack = self.shared.icon_button(
            "minus", self._dom_remove_stack_click
        )
        self._dom_cats = Select(options=[*self._categories], align="center")
        self._dom_cats_selected = self.shared.multichoice()
        self._dom_cats_all = _btn(
            "All",
            CustomJS(
                args=dict(select=self._dom_cats_selected),
                code="select.value = select.options;",
            ),
        )
        self._dom_cats_clear = _btn(
            "Clear",
            CustomJS(
                args=dict(select=self._dom_cats_selected),
                code="select.value = [];",
            ),
        )
        self._dom_cats.js_on_change(
            "value",
            CustomJS(
                args=dict(
                    choices={k: v._choices for k, v in self._categories.items()},
                    category=self._dom_cats,
                    select=self._dom_cats_selected,
                ),
                code="select.options = choices[category.value]; select.value = select.options;",
            ),
        )
        self._controls = [
            self._dom_add_model,
            self._dom_remove_model,
            self._dom_add_stack,
            self._dom_remove_stack,
            self._dom_cats,
            self._dom_cats_all,
            self._dom_cats_clear,
            self._dom_cats_selected,
        ]
        # models
        self._models: list[_KappaModel] = []
        self._dom_models = column(
            sizing_mode="stretch_width", background=UI.color_background
        )
        # stacks
        self._stacks: list[_StackGroup] = []
        self._dom_stacks = column(
            sizing_mode="stretch_width", background=UI.color_background
        )
        # processes
        self._processes: list[str] = []
        # blocks
        self.dom = column(
            row(
                self._dom_freeze,
                _label(text="Model:"),
                self._dom_add_model,
                self._dom_remove_model,
                _label(text="Stack:"),
                self._dom_add_stack,
                self._dom_remove_stack,
                _label(text="Axis:"),
                self._dom_cats,
                self._dom_cats_all,
                self._dom_cats_clear,
                self._dom_cats_selected,
                sizing_mode="stretch_width",
            ),
            self._dom_models,
            self._dom_stacks,
            sizing_mode="stretch_width",
        )

        # initialize
        self._setup_default()

    def add_model(
        self, model: str = "", pattern: str = "", matched: _KappaMatch = None
    ):
        self._models.append(_KappaModel(model=model, pattern=pattern, matched=matched))
        self._dom_models.children.append(self._models[-1].dom)

    def remove_model(self):
        if len(self._models):
            self._models.pop()
            self._dom_models.children.pop()

    def add_stack(self, name: Optional[str] = None, stacks: Iterable[str] = None):
        self._stacks.append(
            _StackGroup(name=name, processes=self._dom_cats_selected, matched=stacks)
        )
        self._dom_stacks.children.append(self._stacks[-1].dom)

    def remove_stack(self):
        if len(self._stacks):
            self._stacks.pop()
            self._dom_stacks.children.pop()

    def _setup_categories(self, category: str):
        self._dom_cats.value = category
        _select = self._dom_cats_selected
        _select.options = self._categories[category]._choices
        _select.value = _select.options

    def _setup_default(self):
        categories = difflib.get_close_matches(
            _PROCESS, self._categories, n=len(self._categories), cutoff=0
        )
        for cat in categories:
            processes = set(self._categories[cat]._choices)
            for k, vs in Datasets.items():
                for v in vs:
                    matched = _KappaMatch(v, processes, k)
                    if matched:
                        self.add_model(k, v, matched)
                        processes -= set(self._models[-1])
            if self._models:
                self._setup_categories(cat)
                break
        if not self._models:
            self._setup_categories(categories[0])
            processes = set(self._categories[self.process]._choices)
        for k, vs in Stacks:
            if set(vs) <= processes:
                self.add_stack(k, vs)
                processes -= set(self._stacks[-1])

    def _dom_freeze_click(self):
        self.frozen = not self.frozen

    def _dom_add_model_click(self):
        self.add_model()

    def _dom_remove_model_click(self):
        self.remove_model()

    def _dom_add_stack_click(self):
        self.add_stack()

    def _dom_remove_stack_click(self):
        self.remove_stack()

    @property
    def process(self):
        return self._dom_cats.value

    @property
    def selected(self):
        return self._dom_cats_selected.value

    @property
    def frozen(self):
        return self._frozen

    @frozen.setter
    def frozen(self, value: bool):
        self._frozen = value
        if self._frozen:
            processes = set(self._dom_cats_selected.value)
            for model in self._models:
                model.update(processes)
                processes -= set(model)
            for stacks in self._stacks:
                stacks.update(processes)
                processes -= set(stacks)
            self._processes = sorted(processes)

        for control in chain(self._controls, self._models, self._stacks):
            control.disabled = self._frozen
        for k, v in self._FREEZE[self._frozen].items():
            setattr(self._dom_freeze, k, v)
        for callback in self._callbacks:
            callback(self._frozen)

    def render(
        self,
        data: dict[str, Hist1D],  # TODO: plot 2D histogram
        logger: Callable[[str], None],
        log_y: bool,
        **_,
    ):
        y_axis_type = "log" if log_y else "linear"
        # preprocessing
        models: dict[str, list[_KappaModel]] = defaultdict(list)
        for m in self._models:
            if m.matched:
                models[m.model].append(m)
        stacks: list[_StackGroup] = []
        for s in self._stacks:
            if s.matched:
                stacks.append(s)
        # render style options
        glyph_options: dict[str, dict[str, Checkbox]] = defaultdict(dict)
        glyph_option_doms = []
        for k in ("Regular", "Stacked"):
            _options = [Div(text=f"{k}:", styles={"font-weight": "bold"})]
            for label in ("Fill", "Step", "Errorbar"):
                glyph_options[k][label] = Checkbox(label=label, active=True)
                _options.append(glyph_options[k][label])
            glyph_option_doms.append(
                row(
                    *_options, sizing_mode="stretch_width", styles={"border": UI.border}
                )
            )

        # render coupling sliders
        coupling_sliders: dict[str, Slider] = {}
        coupling_doms = []
        for model in self._models:
            if not model.matched:
                continue
            for k in model.matched._model.diagrams[0]:
                if k in coupling_sliders:
                    continue
                slider, dom = self._render_slider(k)
                coupling_sliders[k] = slider
                coupling_doms.append(dom)

        # render legend
        legends: dict[str, Checkbox] = {}
        legends_all = Checkbox(label="All", active=True)
        legend_doms = []

        def legend_add(field: str, label: str):
            legends[field] = Checkbox(label=label, active=True)
            legends_all.js_link("active", legends[field], "active")
            legend_doms.append(legends[field])

        def legend_title(title: str, *badges: str):
            div = Div(text=title, align="center", styles={"font-size": "1.2em"})
            if badges:
                div = self.shared.with_badge(div)
                div.text = "".join(chain(badges, (div.text,)))
            legend_doms.append(div)

        def legend_hr():
            legend_doms.append(self.shared.hr())

        legend_title("Legend")
        for i, p in enumerate(self._processes):
            legend_add(SourceID.regular(i), p)
        legend_hr()
        for k, ms in models.items():
            legend_title(k, m.badge)
            for i, m in enumerate(ms):
                legend_add(SourceID.model(k, i), m.name)
            legend_hr()
        for i, s in enumerate(stacks):
            legend_title(f"{s.name or i+1}", s.badge)
            for j, p in enumerate(s):
                legend_add(SourceID.stack(i, j), p)
            legend_hr()

        # setup colors
        colors: dict[str, RGB] = {}
        for (k, checkbox), color in zip(legends.items(), _palette(len(legends))):
            colors[k] = RGB(color)
            checkbox.styles = {
                "background-color": color,
                "color": colors[k].contrast_best(_WHITE, _BLACK).rgb,
            }

        # TODO render plots
        plots = []
        for title, (val, var, axis) in data.items():
            logger(f'Rendering histogram "{title}"')
            # construct figure
            tooltip = HoverTool(
                tooltips=[
                    ("x\u00B1\u03C3", "@$name"),
                    ("bin", "@label"),
                    ("dataset", "@$name"),
                ],
                point_policy="follow_mouse",
            )
            tooltip.renderers = []
            fig = self.__ticks(
                figure(
                    height=UI.height_figure,
                    sizing_mode="stretch_width",
                    tools="pan,wheel_zoom,box_zoom,reset",
                    y_axis_type=y_axis_type,
                ),
                axis,
            )
            fig.add_tools(tooltip)
            plots.extend((Div(text=title), fig))

            # initialize data
            _left, _right = self.__edges(axis)
            val_edges = {"left": _left, "right": _right, "label": BHAxis.labels(axis)}
            var_edges = {"center": (_left + _right) / 2}
            _bottom = self.__logy_find_bottom(val) if log_y else 0

            # render regular histograms
            val_source = ColumnDataSource(data=val_edges)
            var_source = ColumnDataSource(data=var_edges)
            for i, p in enumerate(self._processes):
                # data source
                field = SourceID.regular(i)
                _val, _err = val[p], np.sqrt(var[p])
                val_source.data[p] = self.__tooltips(_val, _err)
                val_source.data[field] = self.__logy_set_bottom(
                    _val, _bottom if log_y else None
                )
                var_source.data[f"upper_{field}"] = self.__logy_set_bottom(
                    _val + _err, _bottom if log_y else None
                )
                var_source.data[f"lower_{field}"] = self.__logy_set_bottom(
                    _val - _err, _bottom if log_y else None
                )
                # fill glyph
                fill = fig.quad(
                    top=field,
                    bottom=_bottom,
                    fill_color=colors[field].copy(0.1).rgba,
                    source=val_source,
                    **_HIST_FILL,
                    name=p,
                )
                checkboxes = [glyph_options["Regular"]["Fill"], legends[field]]
                fill_visible = CustomJS(
                    args=dict(glyphs=[fill], checks=checkboxes),
                    code=_CODES["visibility"],
                )
                for checkbox in checkboxes:
                    checkbox.js_on_change("active", fill_visible)
                tooltip.renderers.append(fill)
                # step glyph
                step_left = fig.step(
                    y=field,
                    line_color=colors[field].rgba,
                    source=val_source,
                    **_HIST_STEP_LEFT,
                )
                step_right = fig.step(
                    y=field,
                    line_color=colors[field].rgba,
                    source=val_source,
                    **_HIST_STEP_RIGHT,
                )
                checkboxes = [glyph_options["Regular"]["Step"], legends[field]]
                step_visible = CustomJS(
                    args=dict(glyphs=[step_left, step_right], checks=checkboxes),
                    code=_CODES["visibility"],
                )
                for checkbox in checkboxes:
                    checkbox.js_on_change("active", step_visible)
                # errorbar glyph
                errorbar = Whisker(
                    upper=f"upper_{field}",
                    lower=f"lower_{field}",
                    source=var_source,
                    line_color=colors[field].rgba,
                    **_HIST_ERRORBAR,
                )
                fig.add_layout(errorbar)
                checkboxes = [glyph_options["Regular"]["Errorbar"], legends[field]]
                errorbar_visible = CustomJS(
                    args=dict(glyphs=[errorbar], checks=checkboxes),
                    code=_CODES["visibility"],
                )
                for checkbox in checkboxes:
                    checkbox.js_on_change("active", errorbar_visible)

            # TODO render models

            # TODO render stacks

        return column(
            row(*glyph_option_doms, sizing_mode="stretch_width"),
            column(*coupling_doms, sizing_mode="stretch_width"),
            row(
                ScrollBox(
                    child=column(plots, sizing_mode="stretch_both"),
                    sizing_mode="stretch_both",
                ),
                column(
                    row(
                        legends_all,
                        sizing_mode="stretch_width",
                        styles=_BOX_STYLE,
                    ),
                    ScrollBox(
                        child=column(
                            *legend_doms,
                            width=UI.width_side,
                            sizing_mode="stretch_height",
                        ),
                        styles=_BOX_STYLE,
                        sizing_mode="stretch_height",
                    ),
                    sizing_mode="stretch_height",
                ),
                sizing_mode="stretch_both",
            ),
            sizing_mode="stretch_both",
        )

    def _render_slider(self, coupling: str):
        start, end, step = CouplingScan[coupling if coupling in CouplingScan else None]
        slider = Slider(
            start=start,
            end=end,
            step=step,
            value=1,
            title=coupling,
            sizing_mode="stretch_width",
        )
        imin = self.shared.float_input(value=start, high=end, title="Min")
        imax = self.shared.float_input(value=end, low=start, title="Max")
        istep = self.shared.float_input(value=step, low=0, title="Step")
        ivalue = self.shared.float_input(value=1, low=start, high=end, title="Value")

        imin.js_link("value", slider, "start")
        imin.js_link("value", ivalue, "low")
        imin.js_link("value", imax, "low")
        imax.js_link("value", slider, "end")
        imax.js_link("value", ivalue, "high")
        imax.js_link("value", imin, "high")
        ivalue.js_link("value", slider, "value")
        slider.js_link("value", ivalue, "value")
        istep.js_link("value", slider, "step")
        return slider, row(
            [ivalue, slider, istep, imin, imax],
            sizing_mode="stretch_width",
            styles=_BOX_STYLE,
        )

    @staticmethod
    def __ticks(fig: figure, edge: AxesMixin):
        fig.xaxis.axis_label = (edge.label or edge.name).replace("$", "$$")
        fig.yaxis.axis_label = "Events"
        if isinstance(edge, (Regular, Variable)):
            fig.xaxis.ticker = [*edge.edges]
        else:
            labels = BHAxis.labels(edge)
            fig.xaxis.ticker = [*range(len(labels))]
            fig.xaxis.major_label_overrides = dict(enumerate(labels))
        return fig

    @staticmethod
    def __edges(edge: AxesMixin):
        if isinstance(edge, (Regular, Variable)):
            edges = np.asarray(BHAxis.edges(edge, finite=True))
            return edges[:-1], edges[1:]
        else:
            edges = np.arange(len(edge) + sum(BHAxis.flow(edge)))
            return edges - 0.5, edges + 0.5

    @staticmethod
    def __tooltips(val: pd.Series, err: pd.Series):
        return [f"{_FF(v)} \u00B1 {_FF(e)}" for v, e in zip(val, err)]

    @staticmethod
    def __logy_find_bottom(val: pd.DataFrame):
        return 10 ** np.floor(np.log10(val[val > 0].min().min()))

    @staticmethod
    def __logy_set_bottom(val: pd.DataFrame, bottom: Optional[float]):
        return val.clip(lower=bottom) if bottom is not None else val
