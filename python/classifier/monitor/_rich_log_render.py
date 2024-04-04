# modified from https://github.com/Textualize/rich/blob/v13.5.3/rich/_log_render.py
from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING, Iterable, List, Optional, Union

from rich._log_render import FormatTimeCallable
from rich._log_render import LogRender as _LogRender
from rich.containers import Renderables
from rich.table import Table
from rich.text import Text, TextType

if TYPE_CHECKING:
    from rich.console import Console, ConsoleRenderable, RenderableType
    from rich.logging import RichHandler


class LogRender(_LogRender):
    _url_pattern = re.compile(r"^([\w]+)://.*$")
    _https = frozenset({"https", "http"})

    @classmethod
    def _parse_link(cls, path: str, no: str):
        if path:
            if (match := cls._url_pattern.match(path)) is None:
                path = f"file://{path}"
            elif no and (match.groups()[0] in cls._https):
                no = f"L{no}"
        return path, no

    def __call__(
        self,
        console: Console,
        renderables: Iterable["ConsoleRenderable"],
        log_time: Optional[datetime] = None,
        time_format: Optional[Union[str, FormatTimeCallable]] = None,
        level: TextType = "",
        path: Optional[str] = None,
        line_no: Optional[int] = None,
        link_path: Optional[str] = None,
    ) -> Table:

        output = Table.grid(padding=(0, 1))
        output.expand = True
        if self.show_time:
            output.add_column(style="log.time")
        if self.show_level:
            output.add_column(style="log.level", width=self.level_width)
        output.add_column(ratio=1, style="log.message", overflow="fold")
        if self.show_path and path:
            output.add_column(style="log.path")
        row: List["RenderableType"] = []
        if self.show_time:
            log_time = log_time or console.get_datetime()
            time_format = time_format or self.time_format
            if callable(time_format):
                log_time_display = time_format(log_time)
            else:
                log_time_display = Text(log_time.strftime(time_format))
            if log_time_display == self._last_time and self.omit_repeated_times:
                row.append(Text(" " * len(log_time_display)))
            else:
                row.append(log_time_display)
                self._last_time = log_time_display
        if self.show_level:
            row.append(level)

        row.append(Renderables(renderables))
        if self.show_path and path:
            link_path, line_no = self._parse_link(link_path, line_no)
            path_text = Text()
            path_text.append(path, style=f"link {link_path}" if link_path else "")
            if line_no:
                path_text.append(":")
                path_text.append(
                    f"{line_no}",
                    style=f"link {link_path}#{line_no}" if link_path else "",
                )
            row.append(path_text)

        output.add_row(*row)
        return output


def replace_log_render(handler: RichHandler, **kwargs):
    handler._log_render = LogRender(**kwargs)
    return handler
