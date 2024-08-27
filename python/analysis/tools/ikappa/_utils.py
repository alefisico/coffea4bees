import html
from collections import defaultdict
from datetime import datetime
from functools import partial
from glob import glob

from bokeh.document import Document
from bokeh.layouts import row
from bokeh.models import (
    AutocompleteInput,
    Button,
    CustomJS,
    Div,
    InlineStyleSheet,
    Styles,
    TablerIcon,
)

from .config import UI, XRootD


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


class BokehLog:
    def __init__(self, doc: Document, max_history: int = None):
        self._doc = doc

        self._style_ibtn = InlineStyleSheet(css=".bk-btn {padding: 0 6px;}")
        self._dom_log = Div(
            text="",
            height=UI.log_height,
            sizing_mode="stretch_width",
            styles=Styles(
                overflow="auto",
                background=UI.background,
                padding_left="10px",
                margin="0px",
                border="1px solid #C8C8C8",
            ),
        )
        self._dom_dump = Button(
            label="Dump",
            button_type="primary",
            sizing_mode="stretch_height",
            margin=0,
        )
        self._dom_dump.js_on_click(
            CustomJS(
                args=dict(log=self._dom_log),
                code="""
const page = "<!DOCTYPE html><html><head><title>Log</title></head><body>" + log.text + "</body></html>";
const blob = new Blob([page], {type: 'text/plain'});
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = 'log.html';
a.click();
URL.revokeObjectURL(url);
""",
            )
        )
        self.dom = row(self._dom_log, self._dom_dump, sizing_mode="stretch_width")

        self._max = max_history
        self._histories = []

    def __call__(self, *msgs: str, escape=False):
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
        self._doc.add_next_tick_callback(
            partial(self._dom_update_log, "<br>".join(lines))
        )

    def _dom_update_log(self, msg):
        self._histories.insert(
            0, f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        )
        if self._max and len(self._histories) > self._max:
            self._histories = self._histories[: self._max]
        self._dom_log.text = "<br>".join(self._histories)

    def ibtn(self, symbol: str, *onclick, **kwargs):
        btn = Button(
            label="",
            icon=TablerIcon(icon_name=symbol, size="1.5em"),
            aspect_ratio=1,
            button_type="primary",
            align="center",
            stylesheets=[self._style_ibtn],
            **kwargs,
        )
        for click in onclick:
            btn.on_click(click)
        return btn
