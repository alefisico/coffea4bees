from classifier.task import ArgParser, EntryPoint, main
from rich import print

from .help import _print_mod


class Main(main.Main):
    _no_monitor = True
    _no_load = True

    argparser = ArgParser(
        prog="expand",
        description="Expand options from predefined workflows.",
        workflow=[
            ("main", f"read and parse workflows"),
        ],
    )
    argparser.add_argument(
        "files",
        metavar="FILE",
        nargs="+",
        help="list of files to expand",
    )
    argparser.add_argument(
        "--as-template",
        action="store_true",
        help="expand as template",
    )

    def run(self, parser: EntryPoint):
        if self.opts.as_template:
            parser._expand(
                *self.opts.files[1:], fetch_main=True, formatter=self.opts.files[0]
            )
        else:
            parser._expand(*self.opts.files, fetch_main=True)
        print(_print_mod(None, parser.args["main"][0], parser.args["main"][1]))
        for cat in parser._keys:
            for mod, opts in parser.args[cat]:
                print(_print_mod(cat, mod, opts))
