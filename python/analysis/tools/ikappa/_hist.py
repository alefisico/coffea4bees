from __future__ import annotations

import difflib
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import chain
from typing import TYPE_CHECKING, Callable, Generator, Iterable, NamedTuple, Optional

from base_class.physics import di_higgs
from base_class.physics.di_higgs import Coupling, Diagram
from bokeh.layouts import column, grid, row
from bokeh.models import (
    AutocompleteInput,
    Button,
    Checkbox,
    Div,
    Dropdown,
    MultiChoice,
    Row,
    ScrollBox,
    Select,
    Slider,
    TextInput,
)
from hist.axis import AxesMixin, StrCategory

from ._utils import Component
from .config import UI, CouplingScan, Datasets, Stacks

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


class SourceID:
    raw = "_{}".format
    process = "rp{:d}".format
    model = "m{}p{:d}".format
    stack = "s{:d}p{:d}".format


def _label(text: str):
    return Div(text=text, align="center")


def _btn(label: str, *onclick):
    btn = Button(label=label, button_type="primary", align="center")
    for click in onclick:
        btn.on_click(click)
    return btn


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
        self._dom_model.on_change("value", self._dom_model_change)
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

    def _dom_model_change(self, attr, old, new):
        self._dom_patterns.completions = [*Datasets.get(new, ())]

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
        self._dom_cats_all = _btn("All", self._dom_cat_select_all)
        self._dom_cats_clear = _btn("Clear", self._dom_cats_select_none)
        self._dom_cats_selected = self.shared.multichoice()
        self._dom_cats.on_change("value", self._dom_cats_update)
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
                self._dom_cats.value = cat
                break
        if not self._models:
            self._dom_cats.value = categories[0]
            processes = set(self._categories[self.process]._choices)
        for k, vs in Stacks:
            if set(vs) <= processes:
                self.add_stack(k, vs)
                processes -= set(self._stacks[-1])

    def _dom_freeze_click(self):
        self.frozen = not self.frozen

    def _dom_cats_update(self, attr, old, new):
        self._dom_cats_selected.options = self._categories[self.process]._choices
        self._dom_cat_select_all()

    def _dom_cat_select_all(self):
        self._dom_cats_selected.value = self._dom_cats_selected.options

    def _dom_cats_select_none(self):
        self._dom_cats_selected.value = []

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
    ):
        colors: dict[str, str] = {}

        # preprocessing
        models: dict[str, list[_KappaModel]] = defaultdict(list)
        for m in self._models:
            if m.matched:
                models[m.model].append(m)
        stacks: list[_StackGroup] = []
        for s in self._stacks:
            if s.matched:
                stacks.append(s)

        # render coupling sliders
        coupling_link: dict[str, Slider] = {}
        coupling_dom = []
        for model in self._models:
            if not model.matched:
                continue
            for k in model.matched._model.diagrams[0]:
                if k in coupling_link:
                    continue
                slider, dom = self._render_slider(k)
                coupling_link[k] = slider
                coupling_dom.append(dom)

        # render legend # TODO
        legends: dict[str, Checkbox] = {}
        legend_column = []

        def legend_add(field: str, label: str):
            colors[field] = "yellow"  # TODO placeholder
            legends[field] = Checkbox(label=label, active=True)  # TODO placeholder
            legend_column.append(legends[field])

        def legend_title(title: str):
            legend_column.append(
                Div(text=title, align="center", styles={"font-size": "1.2em"})
            )

        def legend_hr():
            legend_column.append(self.shared.hr())

        legend_title("Legend")
        for i, p in enumerate(self._processes):
            legend_add(SourceID.process(i), p)
        legend_hr()
        for k, ms in models.items():
            legend_title(f"(Model) {k}")
            for i, m in enumerate(ms):
                legend_add(SourceID.model(k, i), m.name)
            legend_hr()
        for i, s in enumerate(stacks):
            legend_title(f"(Stack) {s.name or i+1}")
            for j, p in enumerate(s):
                legend_add(SourceID.stack(i, j), p)
            legend_hr()

        # render plots # TODO
        plot_grid = grid(sizing_mode="stretch_both")
        # TODO add hist
        for k, (x, w, axes) in data.items():
            logger(f'Rendering histogram "{k}"')
            print(k, x, w, axes)

        return column(
            column(*coupling_dom, sizing_mode="stretch_width"),
            row(
                ScrollBox(child=plot_grid, sizing_mode="stretch_both"),
                ScrollBox(
                    child=column(
                        *legend_column,
                        width=UI.width_side,
                        sizing_mode="stretch_height",
                    ),
                    styles={"border": UI.border},
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
            ivalue, slider, istep, imin, imax, sizing_mode="stretch_width"
        )
