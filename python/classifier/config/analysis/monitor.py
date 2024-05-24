import json

import fsspec
from classifier.task import Analysis, ArgParser

from ..setting import IO


class Usage(Analysis):
    argparser = ArgParser()
    argparser.add_argument(
        "--input",
        help="the path to usage data in JSON format. If not provided, the current usage data will be used.",
        default=None,
    )

    def analyze(self, _=None):
        path = self.opts.input
        if path is None:
            from classifier.monitor.usage import Usage as _Usage

            data = _Usage._serialize()
            if data is None:
                return []
        else:
            with fsspec.open(self.opts.input) as f:
                data = json.load(f)
        return [_usage_report(data)]


class _usage_report:
    def __init__(self, data: dict):
        self._data = data

    def __call__(self):
        from classifier.monitor.usage.analyze import generate_report

        generate_report(self._data, IO.report / "usage")
