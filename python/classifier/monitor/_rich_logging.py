from __future__ import annotations

import logging
import re

from rich._log_render import LogRender as _LogRender
from rich.logging import RichHandler as _RichHandler
from rich.table import Table
from rich.text import Text

from ..config.state import RepoInfo


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
        *args,
        path: str = None,
        line_no: str = None,
        link_path: str = None,
        **kwargs,
    ) -> Table:
        table = super().__call__(
            *args, **kwargs, path=path, line_no=line_no, link_path=link_path
        )
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
            table.columns[-1]._cells[-1] = path_text
        return table


class RichHandler(_RichHandler):
    def __init__(
        self,
        *args,
        show_time: bool = True,
        omit_repeated_times: bool = True,
        show_level: bool = True,
        show_path: bool = True,
        log_time_format: str = "[%x %X]",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._log_render = LogRender(
            show_time=show_time,
            show_level=show_level,
            show_path=show_path,
            time_format=log_time_format,
            omit_repeated_times=omit_repeated_times,
            level_width=None,
        )

    def emit(self, record: logging.LogRecord):
        if LogRender._url_pattern.match(record.pathname) is None:
            record.pathname = RepoInfo.get_url(record.pathname)
        return super().emit(record)
