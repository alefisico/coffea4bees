from __future__ import annotations

import difflib
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from html import escape
from itertools import chain, cycle
from typing import (
    TYPE_CHECKING,
    Callable,
    Collection,
    Generator,
    Iterable,
    Literal,
    NamedTuple,
    Optional,
)

import numpy as np
import numpy.typing as npt
from base_class.physics import di_higgs
from base_class.physics.di_higgs import Coupling, Diagram
from bokeh.layouts import column, row
from bokeh.models import (
    AutocompleteInput,
    Button,
    Checkbox,
    ColumnDataSource,
    CustomJS,
    CustomJSHover,
    Div,
    Dropdown,
    HoverTool,
    MultiChoice,
    Renderer,
    Row,
    ScrollBox,
    Select,
    Slider,
    TablerIcon,
    TextInput,
    Whisker,
)
from bokeh.plotting import figure
from hist.axis import (
    Regular,
    StrCategory,
    Variable,
)

from ._bh import BHAxis, HistAxis
from ._utils import RGB, Component
from .config import UI, Plot
from .preset import (
    CouplingScan,
    ModelPatterns,
    Palette,
    StackGroups,
    VisibleGlyphs,
)

_VisibleGlyphs = set(VisibleGlyphs)

if TYPE_CHECKING:
    import pandas as pd

    from ._plot import AxisProjector
    from ._widget import ClickableDiv

    class Hist1D(NamedTuple):
        values: pd.DataFrame
        variances: pd.DataFrame
        edge: HistAxis

    class Hist2D(NamedTuple):
        values: pd.DataFrame
        variances: pd.DataFrame
        edges: tuple[HistAxis, HistAxis]


# constants
_DIHIGGS = sorted(set(di_higgs.__all__) - {"Coupling"})
_WHITE = RGB(255, 255, 255)
_BLACK = RGB(0, 0, 0)

# styles
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

_HIST_ERRORBAR_BAR = dict(
    upper_head=None,
    lower_head=None,
    base="center",
    level="annotation",
    line_width=1,
)

_HIST_ERRORBAR_DOT = dict(
    x="center",
    size=5,
    marker="circle",
    line_width=0,
)

_BOX_STYLE = {
    "border": UI.border,
    "margin-top": "-1px",
}

_GLYPH_ICONS = {
    "fill": "chart-bar",
    "step": "chart-line",
    "errorbar": "chart-candle",
}


# functions
_FF = f"{{:.{Plot.tooltip_float_precision}g}}".format


def _find_process(categories: Collection[str]):
    return difflib.get_close_matches("process", categories, n=len(categories), cutoff=0)


# html & js
BADGE = '<span class="badge" style="background-color: {color};">{text}</span>'.format

_CODES = {
    "py_float_format": """
function pyFloatFormat(value, precision) {
    return value.toPrecision(precision).replace(/((\.0+)|((?<=\.\d*?)0+))(?=(e|$))/, "").replace(/e([+-])(\d)$/, "e$10$2");
}
""",
    "preprocess": """
function preprocess(val, err, normalize, density, width) {
    const total = val.reduce((a, b) => a + b, 0);
    const label_sum = `${pyFloatFormat(total, precision)} \u00B1 ${pyFloatFormat(Math.sqrt(err.reduce((a, b) => a + b, 0)), precision)}`;
    err = err.map(Math.sqrt);
    let label_bin = new Array(val.length);
    for (let i = 0; i < val.length; i++) {
        label_bin[i] = `${pyFloatFormat(val[i], precision)} \u00B1 ${pyFloatFormat(err[i], precision)}`;
    }
    if (normalize) {
        const norm = Math.abs(total);
        val = val.map(v => v / norm);
        err = err.map(e => e / norm);
    }
    if (density && (width !== null)) {
        val = val.map((v, i) => v / width[i]);
        err = err.map((e, i) => e / width[i]);
    }
    if (density || normalize) {
        for (let i = 0; i < label_bin.length; i++) {
            label_bin[i] = `${label_bin[i]} (${pyFloatFormat(val[i], precision)} \u00B1 ${pyFloatFormat(err[i], precision)})`;
        }
    }
    return [[val, err], [label_sum], label_bin];
}
""",
    "logy_set_bottom": """
function logySetBottom(val, bottom, log_y) {
    return log_y ? val.map(v => v < bottom ? bottom : v) : val;
}
""",
}


class _DataField:
    regular = "r_{:d}".format
    model = "m_{}__p_{:d}".format
    basis = "{}__b_{:d}".format
    stack = "s_{:d}__p_{:d}".format

    raw = "raw_{}".format
    var = "var_{}".format


class _PlotField:
    upper = "upper_{}".format
    lower = "lower_{}".format
    bottom = "bottom_{}".format
    label_count = "label_count_{}".format


class _ColumnLike:
    def __init__(self, val: pd.DataFrame):
        test = val[val.columns[0]]
        self._shape = len(test)
        self._dtype = test.dtype

    def zeros(self) -> npt.NDArray:
        return np.zeros(self._shape, self._dtype)

    def copy(self, arr: npt.NDArray) -> npt.NDArray:
        new = self.zeros()
        new[:] = arr
        return new


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


def _checkbox(active: bool = True):
    return Checkbox(
        active=active,
        width=Plot.legend_checkbox_width - 10,  # 5px margin
    )


def _palette(n: int):
    selected = cycle(Palette)
    return [next(selected) for _ in range(n)]


def _link_checkbox(checkbox: Checkbox, plot: Renderer):
    checkbox.js_link("active", plot, "visible")
    if not checkbox.active:
        plot.visible = False


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
            self.model: Diagram = getattr(di_higgs, model)(
                basis=self._couplings, unit_basis_weight=False
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
    badge: str
    index: int
    matched: bool

    dom_hint: list
    dom_hide: list
    dom_matched: Dropdown
    dom: Row

    @abstractmethod
    def __iter__(self): ...

    def _disable(self, value: bool):
        if value:
            matched = [*self]
            self.dom_matched.menu = [(m, m) for m in matched]
            self.dom_matched.label = "Matched" if matched else "No Match"
            self.dom_matched.button_type = "success" if matched else "danger"

            self.dom.children = self.dom_hint + [self.dom_matched]
        else:
            self.dom.children = self.dom_hint + self.dom_hide

    @property
    def _index(self):
        return self.index + 1


class _Model(_Matched):
    badge = BADGE(color=UI.badge_color["model"], text="model")
    matched: _KappaMatch

    def __init__(
        self,
        index: int,
        model: str,
        pattern: str,
        matched: _KappaMatch = None,
    ):
        self.index = index
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
                    patterns={k: list(v) for k, v in ModelPatterns.items()},
                    select=self._dom_model,
                    input=self._dom_patterns,
                ),
                code="input.completions = patterns[select.value] || [];",
            ),
        )
        self._dom_model.value = model

        self.dom_hint = [_label(text=f"Model{self._index}:"), self._dom_model]
        self.dom_hide = [_label(text="Pattern:"), self._dom_patterns]
        self.dom_matched = Dropdown(sizing_mode="stretch_width", align="center")
        self.dom = row(sizing_mode="stretch_width")

        self.disabled = False

    def __iter__(self):
        yield from self.matched._match

    def __contains__(self, item):
        return item in self.matched._match

    def __len__(self):
        return len(self.matched._match)

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
        self._disable(value)

    @property
    def name(self) -> str:
        return self._dom_patterns.value

    @property
    def model(self) -> str:
        return self._dom_model.value

    @property
    def field(self):
        return _DataField.model(self.model, self.index)

    @property
    def fields(self):
        field = self.field
        for i, p in enumerate(self):
            yield _DataField.basis(field, i), p


class _Stack(_Matched):
    badge = BADGE(color=UI.badge_color["stack"], text="stack")
    matched: list[str]

    def __init__(
        self,
        index: int,
        name: str,
        processes: MultiChoice,
        matched: Iterable[str] = None,
    ):
        self.index = index
        self.matched = list(matched or ())

        self._dom_name = TextInput(value=name or "")
        self._dom_bins = MultiChoice(
            value=self.matched,
            options=processes.value,
            sizing_mode="stretch_width",
            align="center",
        )
        self._dom_bins.js_link("options", processes, "value")

        self.dom_hint = [_label(text=f"Stack{self._index}:"), self._dom_name]
        self.dom_hide = [_label(text="Bins:"), self._dom_bins]
        self.dom_matched = Dropdown(sizing_mode="stretch_width", align="center")
        self.dom = row(sizing_mode="stretch_width")

        self.disabled = False

    def __iter__(self) -> Generator[str, None, None]:
        yield from self.matched

    def __contains__(self, item):
        return item in self.matched

    def __len__(self):
        return len(self.matched)

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
        self._disable(value)

    @property
    def name(self):
        return self._dom_name.value or f"Stack{self._index}"

    @property
    def fields(self):
        for i, p in enumerate(self):
            yield _DataField.stack(self.index, i), p


class _Ratio(_Matched):  # TODO add ratio plot
    badge = BADGE(color=UI.badge_color["ratio"], text="ratio")
    matched: tuple[str, str]


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
            self._dom_cats,
            self._dom_cats_all,
            self._dom_cats_clear,
            self._dom_cats_selected,
        ]
        # models
        self._models: list[_Model] = []
        self._dom_models = column(
            sizing_mode="stretch_width", background=UI.background_color
        )
        # stacks
        self._stacks: list[_Stack] = []
        self._dom_stacks = column(
            sizing_mode="stretch_width", background=UI.background_color
        )
        # processes
        self._processes: list[str] = []
        # blocks
        self.dom = column(
            row(
                self._dom_freeze,
                _label(text="Axis:"),
                self._dom_cats,
                self._dom_cats_all,
                self._dom_cats_clear,
                self._dom_cats_selected,
                sizing_mode="stretch_width",
            ),
            row(
                *self._dom_transform("model"),
                *self._dom_transform("stack"),
                *self._dom_transform("ratio"),
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
        self._models.append(
            _Model(
                index=len(self._models),
                model=model,
                pattern=pattern,
                matched=matched,
            )
        )
        self._dom_models.children.append(self._models[-1].dom)

    def remove_model(self):
        if len(self._models):
            self._models.pop()
            self._dom_models.children.pop()

    def add_stack(self, name: Optional[str] = None, stacks: Iterable[str] = None):
        self._stacks.append(
            _Stack(
                index=len(self._stacks),
                name=name,
                processes=self._dom_cats_selected,
                matched=stacks,
            )
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
        categories = _find_process(self._categories)
        for cat in categories:
            processes = set(self._categories[cat]._choices)
            for k, vs in ModelPatterns.items():
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
        for k, vs in StackGroups:
            if set(vs) <= processes:
                self.add_stack(k, vs)
                processes -= set(self._stacks[-1])

    def _dom_freeze_click(self):
        self.frozen = not self.frozen

    def _dom_transform_click(
        self,
        transform: str,
        action: Literal["add", "remove"],
    ):
        getattr(self, f"{action}_{transform}")()

    def _dom_transform(self, transform: str):
        doms = (
            _label(text=f"{transform.capitalize()}:"),
            self.shared.icon_button(
                "plus",
                partial(self._dom_transform_click, transform=transform, action="add"),
            ),
            self.shared.icon_button(
                "minus",
                partial(
                    self._dom_transform_click, transform=transform, action="remove"
                ),
            ),
        )
        self._controls.extend(doms)
        return doms

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
        normalized: bool,
        density: bool,
        log_y: bool,
        flow: bool,
        **_,
    ):
        bhaxis = BHAxis(flow=flow, floating_format=_FF)
        # config
        y_axis_type = "log" if log_y else "linear"
        tags = []
        if normalized:
            tags.append("normalized")
        if density:
            tags.append("density")
        tags = ", ".join(tags)
        if tags:
            tags = f" ({tags})"

        # preprocessing
        models: dict[str, list[_Model]] = defaultdict(list)
        for m in self._models:
            if m.matched:
                models[m.model].append(m)
        stacks: list[_Stack] = []
        for s in self._stacks:
            if s.matched:
                stacks.append(s)

        # palette
        colors: dict[str, RGB] = {}
        palette = iter(
            _palette(
                (
                    len(self._processes)
                    + sum(len(v) for v in chain(models.values(), stacks))
                )
            )
        )

        # render coupling sliders
        coupling_sliders: dict[str, Slider] = {}
        coupling_doms = []
        for model in self._models:
            if not model.matched:
                continue
            for k in model.matched.model.diagrams[0]:
                if k in coupling_sliders:
                    continue
                slider, dom = self._render_slider(k)
                coupling_sliders[k] = slider
                coupling_doms.append(dom)

        # render legend and glyph controls
        legends: dict[str, Checkbox] = {}
        legend_all = _checkbox()
        legend_toggles: dict[str, ClickableDiv] = {}
        glyphs: dict[str, dict[str, Checkbox]] = {}
        glyph_all: dict[str, Checkbox] = {}
        glyph_expand = self.shared.icon_button(
            "chevrons-left", label="glyphs", aspect_ratio=None
        )
        for k in _GLYPH_ICONS:
            glyph_all[k] = _checkbox()
            legend_all.js_link("active", glyph_all[k], "active")

        glyph_doms = [
            row(
                *(
                    TablerIcon(
                        icon_name=v,
                        size=f"{Plot.legend_checkbox_width-6}px",
                        styles={"padding": "0 3px", "title": k},
                    )
                    for k, v in _GLYPH_ICONS.items()
                ),
                width=Plot.legend_glyph_width,
                visible=False,
            ),
            row(*glyph_all.values(), visible=False),
        ]
        legend_doms = []
        legend_title_style = _BOX_STYLE.copy()
        if not coupling_doms:
            legend_title_style.pop("margin-top")

        def legend_add(field: str, label: str, stack: bool = False):
            colors[field] = color = RGB(next(palette))
            legends[field] = legend = _checkbox()
            legend_all.js_link("active", legends[field], "active")
            glyphs[field] = glyph = dict[str, Checkbox]()
            active = False
            for k in _GLYPH_ICONS:
                glyph[k] = _checkbox((label, k) in _VisibleGlyphs)
                active |= glyph[k].active
                legend.js_link("active", glyph[k], "active")
                glyph_all[k].js_link("active", glyph[k], "active")
            if not active:
                for v in glyph.values():
                    v.active = True
            glyph_doms.append(row(*glyph.values(), visible=False))
            title = (self.shared.toggle if stack else Div)(
                text=escape(label),
                sizing_mode="stretch_width",
                styles={
                    "background-color": color.rgb,
                    "color": color.contrast_best(_WHITE, _BLACK).rgb,
                    "border-radius": "4px",
                    "padding": "2px 5px 2px 5px",
                    "word-wrap": "anywhere",
                },
            )
            legend_doms.append(row(glyph_doms[-1], legend, title))
            return title

        def legend_title(title: str, *badges: str):
            padding = Div(
                text="", width=Plot.legend_glyph_width, visible=False, margin=0
            )
            div = Div(text=title, align="start", styles={"font-size": "1.2em"})
            if badges:
                div = self.shared.with_badge(div)
                div.text = "".join(chain(badges, (div.text,)))
            legend_doms.append(row(padding, div, sizing_mode="stretch_width"))
            glyph_doms.append(padding)

        def legend_hr():
            legend_doms.append(self.shared.hr())

        for i, p in enumerate(self._processes):
            legend_add(_DataField.regular(i), p)
        legend_hr()
        for k, ms in models.items():
            legend_title(k, m.badge)
            for m in ms:
                legend_add(m.field, m.name)
            legend_hr()
        for s in stacks:
            legend_title(s.name, s.badge)
            for field, p in s.fields:
                legend_toggles[field] = legend_add(field, p, stack=True)
            legend_hr()

        legend_dom = ScrollBox(
            child=column(*legend_doms, sizing_mode="stretch_height"),
            width=Plot.legend_width,
            styles=_BOX_STYLE,
            sizing_mode="stretch_height",
        )
        glyph_expand.js_on_click(
            CustomJS(
                args=dict(legend=legend_dom, glyphs=glyph_doms, button=glyph_expand),
                code="""
let icon = "chevrons-left";
let visible = false;
let modifier = -1;
if (button.icon.icon_name == "chevrons-left") {
    icon = "chevrons-right";
    visible = true;
    modifier = 1;
}
for (let glyph of glyphs) {
    glyph.visible = visible;
}
button.icon.icon_name = icon;
"""
                + f"""
legend.width = (legend.width + {Plot.legend_glyph_width} * modifier);
""",
            )
        )

        # plot utils
        def setup_source(
            field: str,
            name: str,
            source: ColumnDataSource,
            tooltip: dict[str, ColumnDataSource],
            val: npt.NDArray,
            var: npt.NDArray,
            bottom: float,
            width: list[float] | None,
        ):
            label_count = _PlotField.label_count(name)
            (
                (val, err),
                tooltip["total"].data[label_count],
                source.data[label_count],
            ) = self.__preprocess(val, var, normalized, density, width)
            for k, v in {
                field: val,
                _PlotField.upper(field): val + err,
                _PlotField.lower(field): val - err,
            }.items():
                source.data[k] = self.__logy_set_bottom(v, bottom, log_y)
            return source

        def render_glyphs(
            field: str,
            name: str,
            source: ColumnDataSource,
            bottom: float | str,
            fig: figure,
            renderers: list,
        ):
            color = colors[field]
            # fill glyph
            fill = fig.quad(
                top=field,
                bottom=bottom,
                source=source,
                fill_color=color.copy(Plot.fill_alpha).rgba,
                **_HIST_FILL,
                name=_PlotField.label_count(name),
            )
            _link_checkbox(glyphs[field]["fill"], fill)
            renderers.append(fill)
            # step glyph
            step_left = fig.step(
                y=field,
                source=source,
                line_color=color.rgba,
                **_HIST_STEP_LEFT,
            )
            _link_checkbox(glyphs[field]["step"], step_left)
            step_right = fig.step(
                y=field,
                source=source,
                line_color=color.rgba,
                **_HIST_STEP_RIGHT,
            )
            _link_checkbox(glyphs[field]["step"], step_right)
            # errorbar glyph
            errorbar_bar = Whisker(
                upper=_PlotField.upper(field),
                lower=_PlotField.lower(field),
                source=source,
                line_color=color.rgba,
                **_HIST_ERRORBAR_BAR,
            )
            fig.add_layout(errorbar_bar)
            _link_checkbox(glyphs[field]["errorbar"], errorbar_bar)
            errorbar_dot = fig.scatter(
                y=field,
                source=source,
                fill_color=color.rgba,
                **_HIST_ERRORBAR_DOT,
            )
            _link_checkbox(glyphs[field]["errorbar"], errorbar_dot)

            return fill, step_left, step_right, errorbar_bar, errorbar_dot

        def js_model(
            model: _Model,
            field: str,
            name: str,
            source: ColumnDataSource,
            tooltip: dict[str, ColumnDataSource],
            bottom: float,
            width: list[float] | None,
        ):
            couplings = model.matched.model.diagrams[0]
            sliders = {k: v for k, v in coupling_sliders.items() if k in couplings}
            basis = [f for f, _ in model.fields]
            js_change = CustomJS(
                args=dict(
                    col_field=field,
                    col_name=_PlotField.label_count(name),
                    col_val=[*map(_DataField.raw, basis)],
                    col_var=[*map(lambda x: _DataField.raw(_DataField.var(x)), basis)],
                    col_upper=_PlotField.upper(field),
                    col_lower=_PlotField.lower(field),
                    sliders=sliders,
                    source=source,
                    tooltip=tooltip,
                    bottom=bottom,
                    width=width,
                    normalized=normalized,
                    density=density,
                    log_y=log_y,
                    precision=Plot.tooltip_float_precision,
                ),
                # fmt: off
                code=
_CODES["py_float_format"] +
_CODES["preprocess"] + 
_CODES["logy_set_bottom"] +"""
const weight = (function() {
""" + "\n".join(f'let __{k} = sliders["{k}"].value;' for k in couplings)
+ "\n".join(model.matched.model.js_weight(**{k: f"__{k}" for k in couplings})) + """
    return __w;
})();
const shape = source.data[col_field].length;
let val = new Float64Array(shape);
let err = new Float64Array(shape);
for (let i = 0; i < val.length; i++) {
    for (let j = 0; j < weight.length; j++) {
        val[i] += source.data[col_val[j]][i] * weight[j];
        err[i] += source.data[col_var[j]][i] * (weight[j] ** 2);
    }
}
let label_sum, label_bin;
[[val, err], label_sum, label_bin] = preprocess(val, err, normalized, density, width);
tooltip["total"].data[col_name] = label_sum;
source.data[col_name] = label_bin;
source.data[col_field] = logySetBottom(val, bottom, log_y);
source.data[col_upper] = logySetBottom(val.map((v, i) => v + err[i]), bottom, log_y);
source.data[col_lower] = logySetBottom(val.map((v, i) => v - err[i]), bottom, log_y);
source.change.emit();
""",
            )
            for slider in sliders.values():
                slider.js_on_change("value", js_change)

        def js_stack(
            norm: Optional[float],
            fields: str,
            source: ColumnDataSource,
            bottom: float,
            width: list[float] | None,
        ):
            toggles = [legend_toggles[f] for f in fields]
            js_change = CustomJS(
                args=dict(
                    col_fields=fields,
                    col_val=[*map(_DataField.raw, fields)],
                    col_var=[*map(lambda x: _DataField.raw(_DataField.var(x)), fields)],
                    col_upper=[*map(_PlotField.upper, fields)],
                    col_lower=[*map(_PlotField.lower, fields)],
                    col_bottom=[*map(_PlotField.bottom, fields)],
                    norm=norm,
                    toggles=toggles,
                    source=source,
                    bottom=bottom,
                    width=width,
                    normalized=normalized,
                    density=density,
                    log_y=log_y,
                ),
                # fmt: off
                code=
_CODES["logy_set_bottom"] + """
function _normalize(val){
    val.forEach((_, j) => val[j] /= norm);
}
function _density(val){
    val.forEach((_, j) => val[j] /= width[j]);
}
const enabled = toggles.map(t => !t.disabled);
const shape = source.data[col_fields[0]].length;
const nan = new Array(shape).fill(null);
let vals = new Array(col_fields.length + 1);
vals[0] = new Float64Array(shape);
let errs = vals[vals.length] = new Float64Array(shape);
for (let i = 0; i < col_fields.length; i++) {
    if (enabled[i]) {
        let val = source.data[col_val[i]];
        let err = source.data[col_var[i]];
        vals[i + 1] = vals[i].map((v, j) => v + val[j]);
        errs.forEach((_, j) => errs[j] += err[j]);
    }
    else {
        vals[i + 1] = vals[i].slice();
    }
}
errs.forEach((v, j) => errs[j] = Math.sqrt(v));
if (normalized && norm !== null) {
    vals.forEach(_normalize);
}
if (density && width !== null) {
    vals.forEach(_density);
}
const last = enabled.lastIndexOf(true);
for (let i = 0; i < col_fields.length; i++) {
    if (enabled[i]) {
        let base = logySetBottom(vals[i + 1], bottom, log_y);
        source.data[col_bottom[i]] = logySetBottom(vals[i], bottom, log_y);
        source.data[col_fields[i]] = base;
        if (i !== last) {
            source.data[col_upper[i]] = base;
            source.data[col_lower[i]] = base;
        }
        else {
            source.data[col_upper[i]] = logySetBottom(vals[i + 1].map((v, j) => v + errs[j]), bottom, log_y);
            source.data[col_lower[i]] = logySetBottom(vals[i + 1].map((v, j) => v - errs[j]), bottom, log_y);
        }
    }
    else {
        source.data[col_bottom[i]] = nan;
        source.data[col_fields[i]] = nan;
        source.data[col_upper[i]] = nan;
        source.data[col_lower[i]] = nan;
    }
}
source.change.emit();
"""
                # fmt: on
            )
            for toggle in toggles:
                toggle.js_on_change("disabled", js_change)

        # render plots
        plots = []
        for title, (val, var, axis) in data.items():
            logger(f'Rendering histogram "{title}"')
            # construct figure
            tooltip_source = {"total": ColumnDataSource(data={})}
            tooltip = HoverTool(
                tooltips=[
                    ("value", "@$name"),
                    ("total", "$total{custom}"),
                    ("bin", "@edge"),
                    ("dataset", "$name{custom}"),
                ],
                formatters={
                    "$name": CustomJSHover(
                        code=f"return special_vars.name.slice({len(_PlotField.label_count(''))});"
                    ),
                    "$total": CustomJSHover(
                        args=dict(source=tooltip_source["total"]),
                        code="return source.data[special_vars.name][0];",
                    ),
                },
                point_policy="follow_mouse",
            )
            tooltip.renderers = []
            fig = self.__ticks(
                figure(
                    height=Plot.figure_height,
                    sizing_mode="stretch_width",
                    tools="pan,wheel_zoom,box_zoom,reset",
                    y_axis_type=y_axis_type,
                ),
                axis,
                bhaxis,
            )
            fig.add_tools(tooltip)
            plots.extend((Div(text=title + tags), fig))

            # initialize data
            _left, _right = self.__edges(axis, bhaxis)
            _width = bhaxis.widths(axis)
            _edges = {
                "left": _left,
                "right": _right,
                "center": (_left + _right) / 2,
                "edge": bhaxis.labels(axis),
            }
            _bottom = self.__logy_find_bottom(val) if log_y else 0

            # utils
            _np = _ColumnLike(val)
            _plot = partial(
                render_glyphs, bottom=_bottom, fig=fig, renderers=tooltip.renderers
            )
            _plot_stack = partial(render_glyphs, fig=fig, renderers=tooltip.renderers)
            _setup_source = partial(
                setup_source,
                bottom=_bottom,
                width=_width,
                tooltip=tooltip_source,
            )
            _setup_bottom = partial(self.__logy_set_bottom, bottom=_bottom, log_y=log_y)
            _js_model = partial(
                js_model,
                tooltip=tooltip_source,
                bottom=_bottom,
                width=_width,
            )
            _js_stack = partial(
                js_stack,
                bottom=_bottom,
                width=_width,
            )

            # render regular histograms
            source = ColumnDataSource(data=_edges)
            for i, p in enumerate(self._processes):
                # data source
                field = _DataField.regular(i)
                _plot(
                    field=field,
                    name=p,
                    source=_setup_source(
                        field=field, name=p, source=source, val=val[p], var=var[p]
                    ),
                )

            # render models
            sm = Coupling({"": 1.0})
            for k, ms in models.items():
                for i, m in enumerate(ms):
                    source = ColumnDataSource(data=_edges)
                    weight = m.matched.model.weight(sm)[0]
                    _val, _var = _np.zeros(), _np.zeros()
                    for j, (field, p) in enumerate(m.fields):
                        source.data[_DataField.raw(field)] = val[p]
                        source.data[_DataField.raw(_DataField.var(field))] = var[p]
                        _val += val[p] * weight[j]
                        _var += var[p] * (weight[j] ** 2)
                    field = m.field
                    _plot(
                        field=field,
                        name=m.name,
                        source=_setup_source(
                            field=field, name=m.name, source=source, val=_val, var=_var
                        ),
                    )
                    _js_model(model=m, field=field, name=m.name, source=source)

            # render stacks
            for i, stack in enumerate(stacks):
                source = ColumnDataSource(data=_edges)
                _vals, _err, _total = [_np.zeros()], _np.zeros(), None
                for field, p in stack.fields:
                    source.data[_DataField.raw(field)] = val[p]
                    source.data[_DataField.raw(_DataField.var(field))] = var[p]
                    _vals.append(_vals[-1] + val[p])
                    _err += var[p]
                _err = np.sqrt(_err)
                if normalized:
                    _total = _vals[-1].sum()
                    for v in _vals + [_err]:
                        v /= _total
                if density and _width is not None:
                    for v in _vals + [_err]:
                        v /= _width
                for j, (field, p) in enumerate(stack.fields):
                    base = _vals[j + 1]
                    source.data[_PlotField.bottom(field)] = _setup_bottom(_vals[j])
                    source.data[field] = _setup_bottom(base)
                    if j != len(stack) - 1:
                        upper = lower = source.data[field]
                    else:
                        upper = _setup_bottom(base + _err)
                        lower = _setup_bottom(base - _err)
                    source.data[_PlotField.upper(field)] = upper
                    source.data[_PlotField.lower(field)] = lower
                    label_count = _PlotField.label_count(p)
                    (
                        _,
                        tooltip_source["total"].data[label_count],
                        source.data[label_count],
                    ) = self.__preprocess(
                        val[p], var[p], normalized, density, _width, _total
                    )
                    _plot_stack(
                        field=field,
                        name=p,
                        source=source,
                        bottom=_PlotField.bottom(field),
                    )
                _js_stack(
                    norm=_total,
                    fields=[f for f, _ in stack.fields],
                    source=source,
                )

        return column(
            *coupling_doms,
            row(
                ScrollBox(
                    child=column(plots, sizing_mode="stretch_both"),
                    sizing_mode="stretch_both",
                ),
                column(
                    row(
                        glyph_doms[0],
                        Div(
                            text="Legend",
                            sizing_mode="stretch_width",
                            styles={"font-size": "1.2em"},
                        ),
                        sizing_mode="stretch_width",
                        styles=legend_title_style,
                    ),
                    row(
                        glyph_doms[1],
                        legend_all,
                        glyph_expand,
                        sizing_mode="stretch_width",
                        styles=_BOX_STYLE,
                    ),
                    legend_dom,
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
    def __ticks(fig: figure, edge: HistAxis, bhaxis: BHAxis):
        fig.xaxis.axis_label = (edge.label or edge.name).replace("$", "$$")
        fig.yaxis.axis_label = "Events"
        if isinstance(edge, (Regular, Variable)):
            fig.xaxis.ticker = [*edge.edges]
        else:
            labels = bhaxis.labels(edge)
            fig.xaxis.ticker = [*range(len(labels))]
            fig.xaxis.major_label_overrides = dict(enumerate(labels))
        return fig

    @staticmethod
    def __edges(edge: HistAxis, bhaxis: BHAxis):
        if isinstance(edge, (Regular, Variable)):
            edges = np.asarray(bhaxis.edges(edge, finite=True))
            return edges[:-1], edges[1:]
        else:
            edges = np.arange(len(edge) + sum(bhaxis.flow(edge)))
            return edges - 0.5, edges + 0.5

    @staticmethod
    def __logy_find_bottom(val: pd.DataFrame):
        return 10 ** np.floor(np.log10(val[val > 0].min().min()))

    @staticmethod
    def __logy_set_bottom(val: pd.Series | npt.NDArray, bottom: float, log_y: bool):
        return np.clip(val, bottom, None) if log_y else val

    @staticmethod
    def __preprocess(
        val: npt.NDArray,
        var: npt.NDArray,
        normalize: bool,
        density: bool,
        width: npt.NDArray,
        norm: float = None,
    ):
        total = val.sum()
        label_sum = [f"{_FF(total)} \u00B1 {_FF(np.sqrt(var.sum()))}"]
        err = np.sqrt(var)
        label_bin = [f"{_FF(v)} \u00B1 {_FF(e)}" for v, e in zip(val, err)]
        if normalize:
            norm = np.abs(total) if norm is None else norm
            val = val / norm
            err = err / norm
        if density and width is not None:
            val = val / width
            err = err / width
        if normalize or density:
            label_bin = [
                f"{b} ({_FF(v)} \u00B1 {_FF(e)})"
                for b, v, e in zip(label_bin, val, err)
            ]
        return (val, err), label_sum, label_bin
