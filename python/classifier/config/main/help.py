import inspect
import os
import pkgutil
from pathlib import Path
from textwrap import indent

from classifier.task import task as _task
from rich.console import Console

_INDENT = '  '


def _walk_packages(base):
    base = Path(base)
    for root, _, _ in os.walk(base):
        root = Path(root)
        parts = root.relative_to(base).parts
        if any(_task._is_private(p) for p in parts):
            continue
        for mod in pkgutil.iter_modules([str(root)]):
            if _task._is_private(mod.name):
                continue
            yield '.'.join(parts + (mod.name,))


class Main(_task.Main):
    _keys = ' '.join(f'--{k}' for k in _task.Parser._keys)
    argparser = _task.ArgParser(prog='help')
    argparser.add_argument(
        '--all', action='store_true', help=f'list all available modules for [blue]{_keys}[/blue]')
    argparser.add_argument(
        '--html', help=f'write help to html file')

    def __init__(self):
        self._console = Console(record=True)
        super().__init__()

    def _print(self, *args, **kwargs):
        self._console.print(*args, **kwargs)

    def _print_help(self, task: _task.Task, depth: int = 1):
        self._print(indent(task.help(), _INDENT*depth))

    def run(self, parser: _task.Parser):
        main = parser.args['main']
        self._print('[orange3]\[Usage][/orange3]')
        self._print(' '.join([
            f'{parser.entrypoint} [blue]task[/blue] [yellow]\[args ...][/yellow]',
            *(f'[blue]{k}[/blue] [green]module.class[/green] [yellow]\[args ...][/yellow]' for k in parser._preserved)]))
        self._print(indent(
            f'[blue]task[/blue] = [blue]{"|".join(parser._tasks)}[/blue]', _INDENT))
        self._print(indent(
            f'[green]module.class[/green] = [purple]from[/purple] [green]{_task._CLASSIFIER}.{_task._CONFIG}.\[{"|".join(parser._keys)}].module[/green] [purple]import[/purple] [green]class[/green]', _INDENT))
        self._print('\n[orange3]\[Tasks][orange3]')
        self._print(
            f'[blue]help[/blue]')
        self._print_help(self)
        for task in parser._tasks:
            if task != 'help':
                self._print(f'[blue]{task}[/blue]')
                _, cls = parser._fetch_module(f'{task}.Main', _task._MAIN)
                self._print_help(cls)
        self._print('\n[orange3]\[Options][orange3]')
        self._print(f'[blue]task[/blue] [yellow]{" ".join(main[1])}[/yellow]')
        for cat in parser._keys:
            for imp, opts in parser.args[cat]:
                self._print(
                    f'[blue]--{cat}[/blue] [green]{imp}[/green] [yellow]{" ".join(opts)}[/yellow]')
                modname, clsname = parser._fetch_module_name(imp, cat)
                mod, cls = parser._fetch_module(imp, cat)
                if mod is None:
                    self._print(
                        indent(f'[red]Module "{modname}" not found[/red]', _INDENT))
                elif cls is None:
                    self._print(
                        f'[red]Class "{clsname}" not found in module "{modname}"[/red]')
                else:
                    if not issubclass(cls, _task.Task):
                        self._print(
                            f'[red]Class "{clsname}" is not a subclass of Task[/red]')
                    else:
                        self._print_help(cls)
        if self.opts.all:
            self._print('\n[orange3]\[Modules][/orange3]')
            for cat in parser._keys:
                self._print(f'[blue]--{cat}[/blue]')
                for imp in _walk_packages(f'{_task._CLASSIFIER}/{_task._CONFIG}/{cat}/'):
                    _imp = f'{imp}.*'
                    modname, _ = parser._fetch_module_name(_imp, cat)
                    mod, _ = parser._fetch_module(_imp, cat)
                    tasks = {}
                    if mod is not None:
                        for name, obj in inspect.getmembers(mod, inspect.isclass):
                            if issubclass(obj, _task.Task) and not _task._is_private(name) and obj.__module__ == modname:
                                tasks[name] = obj
                    if tasks:
                        for cls in tasks:
                            self._print(
                                indent(f'[green]{imp}.{cls}[/green]', _INDENT))
                            self._print_help(tasks[cls], 2)
        if self.opts.html is not None:
            self._console.save_html(self.opts.html)
