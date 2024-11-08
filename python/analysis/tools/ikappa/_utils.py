from __future__ import annotations

import html
import json
import re
import traceback
from datetime import datetime
from functools import partial
from glob import glob
from typing import Any, Callable, Optional, TypeVar, overload

from base_class.typetools import check_type
from bokeh.document import Document
from bokeh.events import ButtonClick
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
    UIElement,
)
from bokeh.util.callback_manager import EventCallback

from ._widget import ClickableDiv
from .config import UI, Plot
from .preset import XRootD

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


class ExternalLink:
    _ICON = "external-link"
    _ACTION = """
window.open(url, '_blank');
URL.revokeObjectURL(url);
"""

    def __init__(self, shared: SharedDOM, **kwargs):
        kwargs.pop("icon", None)
        self._icon = TablerIcon(icon_name=self._ICON)
        self._button = Button(icon=self._icon, **kwargs)
        self._div = Div(text="", visible=False)
        self.doms = (self._button, self._div)

        self._button.js_on_click(
            CustomJS(
                args=dict(
                    icon=self._icon,
                    button=self._button,
                    spinner=shared._spinner_icon_css,
                ),
                code="""
icon.icon_name="loader";
const index = icon.stylesheets.indexOf(spinner);
if (index === -1) {
    icon.stylesheets = icon.stylesheets.concat(spinner);
}
button.disabled=true;
""",
            )
        )
        self._icon.on_change("icon_name", self._dom_send_page)
        self._div.js_on_change(
            "text",
            CustomJS(
                args=dict(
                    div=self._div,
                    button=self._button,
                    icon=self._icon,
                    spinner=shared._spinner_icon_css,
                ),
                # fmt: off
                code="""
if (div.text !== "") {
    button.disabled = false;
    const index = icon.stylesheets.indexOf(spinner);
    if (index !== -1) {
        icon.stylesheets = icon.stylesheets.slice(0, index).concat(icon.stylesheets.slice(index+1));
    }
""" + 
  f'icon.icon_name = "{self._ICON}";' + """
    let pages = JSON.parse(div.text);
    div.text = "";
    for (let page of pages) {
        let content = page.page;
        if (page.parser !== "") {
            content = (function(text){
                eval(page.parser);
                return text;
            })(content);
        }
        const blob = new Blob([content], {type: 'text/html'});
        const url = URL.createObjectURL(blob);
""" +       
        self._ACTION + """
    }
}
""",
                # fmt: on
            ),
        )

        self._pages: list[tuple] = []

    def __iter__(self):
        yield from self.doms

    def add_page(self, page: Callable[[], str], parser: Optional[str] = None):
        self._pages.append((page, parser))

    def _generate_page(self, page: tuple) -> dict:
        return {"page": page[0](), "parser": page[1] or ""}

    def _dom_send_page(self, attr, old, new):
        if new != "loader":
            return
        if self._pages:
            pages = []
            for page in self._pages:
                pages.append(self._generate_page(page))
            self._div.text = json.dumps(pages)

    @property
    def disabled(self):
        return self._button.disabled

    @disabled.setter
    def disabled(self, value: bool):
        self._button.disabled = value


class DownloadLink(ExternalLink):
    _ICON = "download"
    _ACTION = """
const a = document.createElement('a');
a.href = url;
a.download = page.file+".html";
a.click();
URL.revokeObjectURL(url);
"""

    def add_page(
        self, file: str, page: Callable[[], str], parser: Optional[str] = None
    ):
        self._pages.append((file, page, parser))

    def _generate_page(self, page: tuple) -> dict:
        return {"file": page[0], "page": page[1](), "parser": page[2] or ""}


class Confirmation:
    def __init__(self, shared: SharedDOM, icon: str, **kwargs):
        kwargs.pop("button_type", None)
        kwargs.pop("visible", None)
        self._dom_action = shared.icon_button(icon, **kwargs)
        self._dom_confirm = shared.icon_button(
            "check", button_type="success", visible=False, **kwargs
        )
        self._dom_cancel = shared.icon_button(
            "x", button_type="danger", visible=False, **kwargs
        )
        args = dict(
            action=self._dom_action,
            confirm=self._dom_confirm,
            cancel=self._dom_cancel,
        )
        self._dom_action.js_on_click(
            CustomJS(
                args=args,
                code="""
confirm.visible = true;
cancel.visible = true;
action.visible = false;
""",
            )
        )
        reset = CustomJS(
            args=args,
            code="""
confirm.visible = false;
cancel.visible = false;
action.visible = true;
""",
        )
        self._dom_confirm.js_on_click(reset)
        self._dom_cancel.js_on_click(reset)

        self.doms = (self._dom_action, self._dom_confirm, self._dom_cancel)

    def __iter__(self):
        yield from self.doms

    def on_click(self, handler: EventCallback):
        self._dom_confirm.on_click(handler)


class BokehLog:
    _BLOB = """
const page = "<!DOCTYPE html><html><head><title>Log</title></head><body><div style='white-space: pre;'>" + log.text + "</div></body></html>";
const blob = new Blob([page], {{type: 'text/html'}});
const url = URL.createObjectURL(blob);
{}
URL.revokeObjectURL(url);
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
            height=UI.log_height,
            sizing_mode="stretch_width",
            styles=Styles(
                overflow="auto",
                background=UI.background_color,
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
                code=self._BLOB.format(
                    """
const a = document.createElement('a');
a.href = url;
a.download = 'log.html';
a.click();
"""
                ),
            )
        )
        self._dom_max = Button(
            icon=TablerIcon(icon_name="external-link", size="2em"),
            **self._BUTTON,
        )
        self._dom_max.js_on_click(
            CustomJS(
                args=dict(log=self._dom_log),
                code=self._BLOB.format("window.open(url, '_blank');"),
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


_ElementT = TypeVar("_ElementT", bound=UIElement)


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
            "height": UI.multichoice_height,
            "sizing_mode": "stretch_width",
        }

        self._float_input_style = {
            "width": UI.numeric_input_width,
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
    background-color: {UI.border_color};
}}
"""
                )
            ],
        }

        self._badge_css = InlineStyleSheet(
            css="""
span.badge {
    font-size: 0.7em;
    font-weight: bold;
    padding: 0.1em 0.5em;
    border-radius: 0.4em;
    color: white;
    margin: 0 0.5em;
}
"""
        )

        self._nonempty_css = InlineStyleSheet(
            css="div.choices__inner {background-color: pink;}"
        )

        self._spinner_icon_css = InlineStyleSheet(
            css="""
span.ti {
    display: inline-block;
    animation: spin 1s infinite linear;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); } 
}
"""
        )

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

    def with_badge(self, element: _ElementT) -> _ElementT:
        element.stylesheets.append(self._badge_css)
        return element

    def nonempty(self, element: _ElementT, empty: Optional[bool] = None) -> _ElementT:
        if not hasattr(element, "value"):
            return element
        stylesheets = [i for i in element.stylesheets if i != self._nonempty_css]
        element.js_on_change(
            "value",
            CustomJS(
                args=dict(
                    element=element,
                    defaults=stylesheets.copy(),
                    empty=self._nonempty_css,
                ),
                code="""
const index = element.stylesheets.indexOf(empty);
if (element.value.length === 0) {
    if (index === -1) {
        element.stylesheets = defaults.concat(empty);
    }
} else if (index !== -1) {
    element.stylesheets = defaults.slice();
}
""",
            ),
        )
        if empty is None:
            empty = not element.value
        if empty:
            stylesheets.append(self._nonempty_css)
        element.stylesheets = stylesheets
        return element

    def toggle(self, **kwargs):
        styles = kwargs.get("styles", {})
        toggle = ClickableDiv(**kwargs)
        toggle.js_on_event(
            ButtonClick,
            CustomJS(
                args=dict(
                    toggle=toggle, styles=styles, color=Plot.legend_disabled_color
                ),
                code="""
toggle.disabled = !toggle.disabled;
if (toggle.disabled) {
    styles = {...styles};
    styles["background-color"] = color;
    styles["color"] = "white";
}
toggle.styles = styles;
""",
            ),
        )
        return toggle


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


class RGB:
    __r: int
    __g: int
    __b: int
    __a: float

    __LUMINANCE = (0.2126, 0.7152, 0.0722)
    __PATTERNS = (
        (
            re.compile(
                r"#(?P<r>[0-9a-fA-F]{2})(?P<g>[0-9a-fA-F]{2})(?P<b>[0-9a-fA-F]{2})"
            ),
            16,
        ),
        (
            re.compile(r"rgb\((?P<r>\d+)(,)?\s*(?P<g>\d+)(,)?\s*(?P<b>\d+)\)"),
            10,
        ),
        (
            re.compile(
                r"rgba\((?P<r>\d+)(,)?\s*(?P<g>\d+)(,)?\s*(?P<b>\d+)\s*([,/])?\s*(?P<a>\d+(\.\d+)?(%)?)\)"
            ),
            10,
        ),
    )

    @overload
    def __init__(self, hex: str, /): ...
    @overload
    def __init__(self, red: int, green: int, blue: int, /): ...
    @overload
    def __init__(self, red: int, greeb: int, blue: int, alpha: float, /): ...
    @overload
    def __init__(self, red: float, green: float, blue: float, /): ...
    @overload
    def __init__(self, red: float, green: float, blue: float, alpha: float, /): ...
    def __init__(self, *color):
        _rgba = None
        if check_type(color, tuple[str]):
            for pattern, base in self.__PATTERNS:
                if match := pattern.fullmatch(color[0]):
                    match = match.groupdict()
                    _rgba = tuple(int(match[c], base) for c in "rgb")
                    if "a" in match:
                        if match["a"].endswith("%"):
                            _rgba += (float(match["a"][:-1]) / 100,)
                        else:
                            _rgba += (float(match["a"]),)
                    break
        elif check_type(color, tuple[int, int, int] | tuple[int, int, int, float]):
            _rgba = color
        elif check_type(
            color, tuple[float, float, float] | tuple[float, float, float, float]
        ):
            _rgba = tuple(int(c * 255) if i < 3 else c for i, c in enumerate(color))
        if _rgba is None:
            raise ValueError(f'Invalid color: "{color}"')
        if len(_rgba) == 3:
            _rgba += (1.0,)
        self.__r, self.__g, self.__b, self.__a = _rgba

    @property
    def __rgb(self):
        return self.__r, self.__g, self.__b

    @property
    def __rgba(self):
        return self.__r, self.__g, self.__b, self.__a

    @property
    def hex(self):
        return "#{:02x}{:02x}{:02x}".format(*self.__rgb)

    @property
    def rgb(self):
        return "rgb({:d}, {:d}, {:d})".format(*self.__rgb)

    @property
    def rgba(self):
        return "rgba({:d}, {:d}, {:d}, {:.6g})".format(*self.__rgba)

    def __lumi(self, c: int):
        c /= 255
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    @property
    def luminance(self):
        return sum(self.__lumi(c) * w for c, w in zip(self.__rgb, self.__LUMINANCE))

    def contrast(self, other: RGB):
        c = (self.luminance + 0.05) / (other.luminance + 0.05)
        return c if c > 1 else 1 / c

    def contrast_best(self, *colors: RGB):
        return max(colors, key=lambda x: self.contrast(x))

    def __repr__(self):
        return self.rgba

    def __check(self, value: Any, name: str, _type: type, _range: tuple[Any, Any]):
        if not isinstance(value, _type):
            raise TypeError(f"{name} must be {_type}")
        if not _range[0] <= value <= _range[1]:
            raise ValueError(f"{name} must be between {_range[0]} and {_range[1]}")

    @property
    def alpha(self):
        return self.__a

    @alpha.setter
    def alpha(self, value: float):
        self.__check(value, "alpha", float, (0.0, 1.0))
        self.__a = value

    @property
    def red(self):
        return self.__r

    @red.setter
    def red(self, value: int):
        self.__check(value, "red", int, (0, 255))
        self.__r = value

    @property
    def green(self):
        return self.__g

    @green.setter
    def green(self, value: int):
        self.__check(value, "green", int, (0, 255))
        self.__g = value

    @property
    def blue(self):
        return self.__b

    @blue.setter
    def blue(self, value: int):
        self.__check(value, "blue", int, (0, 255))
        self.__b = value

    def copy(self, alpha: Optional[float] = None):
        new = RGB(self.red, self.green, self.blue, self.alpha)
        if alpha is not None:
            new.alpha = alpha
        return new
