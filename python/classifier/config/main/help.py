import inspect
import os
import pkgutil
import re
from pathlib import Path
from textwrap import indent

from classifier.task import ArgParser, EntryPoint, Task, main, parse
from classifier.task.special import WorkInProgress
from classifier.task.task import _INDENT

from .. import setting as cfg

_NOTES = [
    f"A special task/flag [blue]{main._FROM}[/blue]/[blue]{main._DASH}{main._FROM}[/blue] [yellow]file \[file ...][/yellow] can be used to load and merge workflows from files. If an option is marked as {parse.EMBED}, it can directly read the jsonable object embedded in the workflow configuration file.",
    f"A special task/flag [blue ]{main._TEMPLATE}[/blue] [yellow]formatter file \[file ...][/yellow] can be used to load and merge workflows and replace the keys by the formatter.",
]


def _print_mod(cat: str, imp: str, opts: list[str | dict], newline: str = "\n"):
    if cat is None:
        output = [f"[blue]{imp}[/blue]"]
    else:
        output = [f"[blue]{main._DASH}{cat}[/blue] [green]{imp}[/green]"]
    current = []
    for opt in opts + [None]:
        if (isinstance(opt, str) and opt.startswith(main._DASH)) or (opt is None):
            if current:
                output.append(indent(f"[yellow]{' '.join(current)}[/yellow]", _INDENT))
            current.clear()
        current.append(opt)
    return newline.join(output)


def _walk_packages(base):
    base = Path(base)
    yield ""
    for root, _, _ in os.walk(base):
        root = Path(root)
        parts = root.relative_to(base).parts
        if any(main._is_private(p) for p in parts):
            continue
        for mod in pkgutil.iter_modules([str(root)]):
            if main._is_private(mod.name):
                continue
            yield ".".join(parts + (mod.name,))


class Main(main.Main):
    _no_state = True

    _keys = " ".join(f"{main._DASH}{k}" for k in EntryPoint._keys)
    argparser = ArgParser(
        prog="help",
        description="Print help information.",
        workflow=[
            ("main", "[blue]task.help()[/blue] print help information"),
        ],
    )
    argparser.add_argument(
        "--all",
        action="store_true",
        help=f"list all available modules for [blue]{_keys}[/blue]",
    )
    argparser.add_argument(
        "--html", action="store_true", help='write "help.html" to output directory'
    )
    argparser.add_argument(
        "--filter",
        type=re.compile,
        help="a Python style regular expression to filter modules when [yellow]--all[/yellow] is set",
        default=".*",
    )
    argparser.add_argument(
        "--wip",
        "--work-in-progress",
        action="store_true",
        help="list tasks that is still [red]Work In Progress[/red]",
        default=False,
    )

    @classmethod
    def prelude(cls):
        cfg.Analysis.enable = False
        cfg.Monitor.enable = False

    def __init__(self):
        super().__init__()
        from rich.console import Console

        self._console = Console(record=True, markup=True)

    def _print(self, *args, **kwargs):
        self._console.print(*args, **kwargs)

    def _print_help(self, task: type[Task], depth: int = 1):
        self._print(indent(task.help(), _INDENT * depth))

    def _check_wip(self, cls: type, depth: int = 0):
        if isinstance(cls, type) and issubclass(cls, WorkInProgress):
            if self.opts.wip:
                self._print(indent("[red]\[Work In Progress][/red]", _INDENT * depth))
            else:
                return False
        return True

    def run(self, parser: EntryPoint):
        import rich.terminal_theme as themes

        tasks = parser.args["main"]
        self._print("[orange3]\[Usage][/orange3]")
        self._print(
            " ".join(
                [
                    f"{parser.entrypoint} [blue]task[/blue] [yellow]\[args ...][/yellow]",
                    *(
                        f"[blue]{main._DASH}{k}[/blue] [green]module.class[/green] [yellow]\[args ...][/yellow]"
                        for k in parser._keys
                    ),
                ]
            )
        )
        self._print(
            indent(
                f'[blue]task[/blue] = [blue]{"|".join(parser._tasks)}[/blue]', _INDENT
            )
        )
        self._print(
            indent(
                f'[green]module.class[/green] = [purple]from[/purple] [green]{main._CLASSIFIER}.{main._CONFIG}.\[{"|".join(parser._keys)}].module[/green] [purple]import[/purple] [green]class[/green]',
                _INDENT,
            )
        )
        self._print("\n[orange3]\[Notes][orange3]")
        for i, note in enumerate(_NOTES, 1):
            self._print(f"#{i}")
            self._print(indent(note, _INDENT))
        self._print("\n[orange3]\[Tasks][orange3]")
        self._print("[blue]help[/blue]")
        self._print_help(self)
        for task in parser._tasks:
            if task != "help":
                _, cls = parser._fetch_module(f"{task}.Main", main._MAIN)
                if self._check_wip(cls):
                    self._print(f"[blue]{task}[/blue]")
                    self._print_help(cls)
        self._print("\n[orange3]\[Options][orange3]")
        self._print(_print_mod(None, "task", tasks[1]))
        for cat in parser._keys:
            target = EntryPoint._keys[cat]
            for imp, opts in parser.args[cat]:
                modname, clsname = parser._fetch_module_name(imp, cat)
                mod, cls = parser._fetch_module(imp, cat)
                self._check_wip(cls)
                self._print(_print_mod(cat, imp, opts))
                if mod is None:
                    self._print(
                        indent(f'[red]Module "{modname}" not found[/red]', _INDENT)
                    )
                elif cls is None:
                    self._print(
                        f'[red]Class "{clsname}" not found in module "{modname}"[/red]'
                    )
                else:
                    if not issubclass(cls, target):
                        self._print(
                            f'[red]Class "{clsname}" is not a subclass of {target.__name__}[/red]'
                        )
                    else:
                        self._print_help(cls)
        if self.opts.all:
            self._print("\n[orange3]\[Modules][/orange3]")
            for cat in parser._keys:
                target = EntryPoint._keys[cat]
                self._print(f"[blue]{main._DASH}{cat}[/blue]")
                for imp in _walk_packages(f"{main._CLASSIFIER}/{main._CONFIG}/{cat}/"):
                    if imp:
                        imp = f"{imp}."
                    _imp = f"{imp}*"
                    modname, _ = parser._fetch_module_name(_imp, cat)
                    mod, _ = parser._fetch_module(_imp, cat)
                    classes = {}
                    if mod is not None:
                        for name, obj in inspect.getmembers(mod, inspect.isclass):
                            if (
                                issubclass(obj, target)
                                and not main._is_private(name)
                                and obj.__module__.startswith(modname)
                            ):
                                fullname = f"{imp}{name}"
                                if self.opts.filter.fullmatch(fullname) is not None:
                                    classes[fullname] = obj
                    if classes:
                        for cls in classes:
                            if self._check_wip(classes[cls], 1):
                                self._print(indent(f"[green]{cls}[/green]", _INDENT))
                                self._print_help(classes[cls], 2)
        if self.opts.html:
            self._console.save_html(cfg.IO.output / "help.html", theme=themes.MONOKAI)
