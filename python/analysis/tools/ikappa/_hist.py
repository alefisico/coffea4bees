from __future__ import annotations

import difflib
import re
from itertools import chain
from typing import TYPE_CHECKING, Callable, Iterable, NamedTuple

from base_class.physics import di_higgs
from base_class.physics.di_higgs import Coupling, Diagram
from bokeh.layouts import column, row
from bokeh.models import (
    AutocompleteInput,
    Button,
    Div,
    Dropdown,
    MultiChoice,
    Select,
    Slider,
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


def _div(text: str):
    return Div(text=text, align="center")


def _btn(label: str, *onclick):
    btn = Button(label=label, button_type="primary", align="center")
    for click in onclick:
        btn.on_click(click)
    return btn


def _update_menu(drop: Dropdown, matched: list[str]):
    drop.menu = [(m, m) for m in matched]
    drop.label = "Matched" if matched else "No Match"
    drop.button_type = "success" if matched else "danger"


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


class _KappaModel:
    def __init__(self, model: str, pattern: str, matched: _KappaMatch = None):
        self.matched = _KappaMatch() if matched is None else matched

        self._dom_model = Select(options=_DIHIGGS, align="center")
        self._dom_pattern_div = _div(text="Pattern:")
        self._dom_patterns = AutocompleteInput(
            value=pattern,
            restrict=False,
            sizing_mode="stretch_width",
            align="center",
            min_characters=0,
        )
        self._dom_model.on_change("value", self._dom_model_change)
        self._dom_model.value = model
        self._dom_matched = Dropdown(sizing_mode="stretch_width", align="center")

        self.dom = row(
            _div(text="Model:"),
            self._dom_model,
            self._dom_pattern_div,
            self._dom_patterns,
            sizing_mode="stretch_width",
        )

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
        if value:
            if self.matched:
                matched = sorted(self)
            else:
                matched = []
            _update_menu(self._dom_matched, matched)
            self.dom.children = self.dom.children[:2] + [self._dom_matched]
        else:
            self.dom.children = self.dom.children[:2] + [
                self._dom_pattern_div,
                self._dom_patterns,
            ]


class _StackGroup:
    def __init__(self, processes: MultiChoice, matched: Iterable[str] = None):
        self.matched = list(matched or ())
        self._dom_stacks = MultiChoice(
            value=self.matched,
            options=processes.value,
            sizing_mode="stretch_width",
            align="center",
        )
        self._dom_stacks.js_link("options", processes, "value")
        self._dom_stacked = Dropdown(sizing_mode="stretch_width", align="center")

        self.dom = row(
            _div(text="Stack:"),
            self._dom_stacks,
            sizing_mode="stretch_width",
        )

        self.disabled = False

    def __iter__(self):
        yield from self._dom_stacks.value

    def __contains__(self, item):
        return item in self._dom_stacks.value

    def update(self, processes: Iterable[str]):
        if set(self._dom_stacks.value) <= set(processes):
            self.matched = self._dom_stacks.value

    @property
    def disabled(self):
        return self._dom_stacks.disabled

    @disabled.setter
    def disabled(self, value: bool):
        self._dom_stacks.disabled = value
        if value:
            if self.matched:
                matched = sorted(self)
            else:
                matched = []
            _update_menu(self._dom_stacked, matched)
            self.dom.children[-1] = self._dom_stacked
        else:
            self.dom.children[-1] = self._dom_stacks


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
        # blocks
        self.dom = column(
            row(
                self._dom_freeze,
                _div(text="Model:"),
                self._dom_add_model,
                self._dom_remove_model,
                _div(text="Stack:"),
                self._dom_add_stack,
                self._dom_remove_stack,
                _div(text="Axis:"),
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

    def add_stack(self, stacks: Iterable[str] = None):
        self._stacks.append(
            _StackGroup(processes=self._dom_cats_selected, matched=stacks)
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
        for ks in Stacks:
            if set(ks) <= processes:
                self.add_stack(ks)
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
        # TODO add hist
        for k, (x, w, axes) in data.items():
            print(k, x, w, axes)
        return column(
            column(*coupling_dom, sizing_mode="stretch_width"),
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
        istep = self.shared.float_input(value=step, low=0, high=1, title="Step [0,1]")
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
            istep, imin, imax, slider, ivalue, sizing_mode="stretch_width"
        )
