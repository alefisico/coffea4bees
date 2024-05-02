import json

import fsspec
from classifier.task import ArgParser, main

from ..setting import IO as cfg


class Main(main.Main):
    _no_monitor = True
    _no_state = True

    argparser = ArgParser(
        prog="report_usage",
        description="Generate an usage report.",
        workflow=[
            ("main", f"analyze and render"),
        ],
    )

    argparser.add_argument(
        "usage",
        help="the path to usage data in JSON format",
    )

    def run(self, _):
        from classifier.monitor.usage.analyze import generate_report

        with fsspec.open(self.opts.usage) as f:
            usage = json.load(f)
        generate_report(usage, cfg.report / "usage")
