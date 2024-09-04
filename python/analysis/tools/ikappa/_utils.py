from __future__ import annotations

import html
import traceback
from datetime import datetime
from functools import partial
from glob import glob
from typing import Callable, Optional

from bokeh.document import Document
from bokeh.layouts import row
from bokeh.models import (
    AutocompleteInput,
    Button,
    CustomJS,
    Div,
    InlineStyleSheet,
    MultiChoice,
    NumericInput,
    Styles,
    TablerIcon,
)
from bokeh.util.callback_manager import EventCallback

from .config import UI, XRootD

_STYLESHEETS = "stylesheets"


def _change_input_glob(self: AutocompleteInput, attr, old, new: str):
    empty = not (paths := glob(f"{new}*"))
    paths.extend(XRootD)
    if empty:
        paths.append(new)
    self.completions = paths


def PathInput(**kwargs):
    kwargs.setdefault("restrict", False)
    kwargs.setdefault("min_characters", 1)
    model = AutocompleteInput(**kwargs)
    model.on_change("value_input", partial(_change_input_glob, model))
    return model


class Confirmation:
    def __init__(self, shared: SharedDOM, icon: str, **kwargs):
        kwargs.pop("button_type", None)
        self._dom_action = shared.icon_button(icon, **kwargs)
        self._dom_confirm = shared.icon_button("check", button_type="success", **kwargs)
        self._dom_cancel = shared.icon_button("x", button_type="danger", **kwargs)
        self._dom_action.on_click(self._dom_action_click)
        self._dom_confirm.on_click(self._dom_reset)
        self._dom_cancel.on_click(self._dom_reset)

        self.dom = row()
        self._dom_reset()

    def _dom_action_click(self):
        self.dom.children = [self._dom_confirm, self._dom_cancel]

    def _dom_reset(self):
        self.dom.children = [self._dom_action]

    def on_click(self, handler: EventCallback):
        self._dom_confirm.on_click(handler)


class BokehLog:
    _BLOB = """
const page = "<!DOCTYPE html><html><head><title>Log</title></head><body><div style='white-space: pre;'>" + log.text + "</div></body></html>";
const blob = new Blob([page], {type: 'text/html'});
const url = URL.createObjectURL(blob);
"""
    _BUTTON = {
        "label": "",
        "button_type": "primary",
        "sizing_mode": "stretch_height",
        "margin": 0,
    }

    def __init__(self, doc: Document, max_history: int = None):
        self._doc = doc
        self._dom_log = Div(
            text="",
            height=UI.height_log,
            sizing_mode="stretch_width",
            styles=Styles(
                overflow="auto",
                background=UI.color_background,
                padding_left="10px",
                margin="0px",
                border=UI.border,
                white_space="pre",
            ),
        )
        self._dom_dump = Button(
            icon=TablerIcon(icon_name="download", size="2em"),
            **self._BUTTON,
        )
        self._dom_dump.js_on_click(
            CustomJS(
                args=dict(log=self._dom_log),
                code=self._BLOB
                + """
const a = document.createElement('a');
a.href = url;
a.download = 'log.html';
a.click();
URL.revokeObjectURL(url);
""",
            )
        )
        self._dom_max = Button(
            icon=TablerIcon(icon_name="window-maximize", size="2em"),
            **self._BUTTON,
        )
        self._dom_max.js_on_click(
            CustomJS(
                args=dict(log=self._dom_log),
                code=self._BLOB
                + """
window.open(url, '_blank');
URL.revokeObjectURL(url);
""",
            )
        )
        self.dom = row(
            self._dom_log, self._dom_dump, self._dom_max, sizing_mode="stretch_width"
        )

        self._max = max_history
        self._histories = []

    def __call__(
        self, *msgs: str, escape=False, transform: Callable[[str], str] = None
    ):
        lines = []
        for msg in msgs:
            if hasattr(msg, "__html__"):
                lines.append(msg.__html__())
            else:
                if not isinstance(msg, str):
                    msg = repr(msg)
                if escape:
                    msg = html.escape(msg)
                lines.append(msg.replace("\n", "<br>"))
        if transform:
            lines = map(transform, lines)
        self._doc.add_next_tick_callback(
            partial(self._dom_update_log, "<br>".join(lines))
        )

    def error(self, *msgs: str, escape=False, exec_info: Exception = None):
        err = []
        if exec_info:
            err.append(
                "".join(
                    traceback.format_exception(
                        type(exec_info), exec_info, exec_info.__traceback__
                    )
                )
            )
        self(
            *msgs,
            *err,
            escape=escape,
            transform=lambda x: f'<font color="red">{x}</font>',
        )

    def _dom_update_log(self, msg):
        self._histories.insert(
            0, f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        )
        if self._max and len(self._histories) > self._max:
            self._histories = self._histories[: self._max]
        self._dom_log.text = "<br>".join(self._histories)


class SharedDOM:
    def __init__(self, doc: Document):
        self._doc = doc

        self._icon_button_style = {
            "label": "",
            "aspect_ratio": 1,
            "button_type": "primary",
            "align": "center",
            _STYLESHEETS: [InlineStyleSheet(css=".bk-btn {padding: 1px 4px;}")],
        }

        self._multichoice_z_index = 1000
        self._multichoice_style = {
            "height": UI.height_multichoice,
            "sizing_mode": "stretch_width",
        }

        self._float_input_style = {
            "width": UI.width_numeric_input,
            "mode": "float",
        }

        self._hr_style = {
            "sizing_mode": "stretch_width",
            _STYLESHEETS: [
                InlineStyleSheet(
                    css=f"""
div.bk-clearfix {{
    width: 100%;
}}
hr {{
    border: none;
    height: 1px;
    background-color: {UI.color_border};
}}
"""
                )
            ],
        }

    @staticmethod
    def __merge(default: dict, custom: dict, **fixed):
        styles = (default, custom, fixed)
        stylesheets = sum((s.get(_STYLESHEETS, []) for s in styles), start=[])
        custom = default | custom | fixed
        if stylesheets:
            custom[_STYLESHEETS] = stylesheets
        return custom

    def multichoice(self, z_index: Optional[int] = None, **kwargs):
        if z_index is None:
            z_index = self._multichoice_z_index
            self._multichoice_z_index += 1
        return MultiChoice(
            **self.__merge(
                self._multichoice_style,
                kwargs,
                stylesheets=[
                    InlineStyleSheet(
                        css=f"div.choices {{background: white;z-index: {z_index};}}"
                    )
                ],
            ),
        )

    def icon_button(self, symbol: str, *onclick, **kwargs):
        btn = Button(
            **self.__merge(
                self._icon_button_style,
                kwargs,
                icon=TablerIcon(icon_name=symbol, size="1.5em"),
            ),
        )
        for click in onclick:
            btn.on_click(click)
        return btn

    def float_input(self, **kwargs):
        return NumericInput(
            **self.__merge(
                self._float_input_style,
                kwargs,
            )
        )

    def hr(self, **kwargs):
        return Div(
            **self.__merge(
                self._hr_style,
                kwargs,
                text="<hr>",
            )
        )


class Component:
    def __init__(
        self, doc: Document = None, log: BokehLog = None, shared: SharedDOM = None
    ):
        self.doc = doc
        self.log = log
        self.shared = shared

    @property
    def inherit_global_states(self):
        return dict(doc=self.doc, log=self.log, shared=self.shared)
