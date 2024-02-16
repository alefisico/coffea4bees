import importlib
import inspect
import os
import pkgutil
from pathlib import Path
from textwrap import indent

from rich import print

from . import task as _task

_INDENT = '  '


def walk_packages(base):
    base = Path(base)
    for root, _, _ in os.walk(base):
        root = Path(root)
        parts = root.relative_to(base).parts
        for mod in pkgutil.iter_modules([str(root)]):
            yield '.'.join(parts + (mod.name,))


class Task(_task._Main):
    argparser = _task.ArgParser(prog='help')
    argparser.add_argument(
        '--sub', action='store_true', default=False, help=f'print help of {_task.Parser._keys}')
    argparser.add_argument(
        '--mods', action='store_true', default=False, help=f'list all available modules in {_task.Parser._keys})')

    @classmethod
    def _print_help(cls, task: _task.Task, depth: int = 1):
        print(indent(task.help(), _INDENT*depth))

    def __init__(self):
        super().__init__()

    def run(self, parser: _task.Parser):
        main = parser.args['main']
        print('[orange3]\[Usage][/orange3]')
        print(' '.join([
            f'{parser.entrypoint} [blue]\[{"|".join(parser._tasks)}][/blue] [yellow]\[args ...][/yellow]',
            *(f'[blue]{k}[/blue] [green]\[module.class][/green] [yellow]\[args ...][/yellow]' for k in parser._preserved)]))
        print(indent(
            f'[green]\[module.class][/green] = [purple]from[/purple] [green]{_task._CLASSIFIER}.{_task._CONFIG}.\[{"|".join(parser._keys)}].module[/green] [purple]import[/purple] [green]class[/green]', _INDENT))
        print('\n[orange3]\[Options][orange3]')
        print(f'[yellow]{" ".join(main[1])}[/yellow]')
        print(
            f'[blue]help[/blue]')
        self._print_help(self)
        for task in parser._tasks:
            if task != 'help':
                print(f'[blue]{task}[/blue]')
                mod = importlib.import_module(f'._{task}', __package__)
                self._print_help(mod.Task)
        for cat in parser._keys:
            for imp, opts in parser.args[cat]:
                print(
                    f'[blue]--{cat}[/blue] [green]{imp}[/green] [yellow]{" ".join(opts)}[/yellow]')
                modname, clsname = parser._fetch_module_name(imp, cat)
                mod, cls = parser._fetch_module(imp, cat)
                if mod is None:
                    print(
                        indent(f'[red]Module "{modname}" not found[/red]', _INDENT))
                elif cls is None:
                    print(
                        f'[red]Class "{clsname}" not found in module "{modname}"[/red]')
                else:
                    if not issubclass(cls, _task.Task):
                        print(
                            f'[red]Class "{clsname}" is not a subclass of Task[/red]')
                    else:
                        if self.opts.sub:
                            self._print_help(cls)
        if self.opts.mods:
            print('\n[orange3]\[Modules][/orange3]')
            for cat in parser._keys:
                print(f'[blue]--{cat}[/blue]')
                for imp in walk_packages(f'{_task._CLASSIFIER}/{_task._CONFIG}/{cat}/'):
                    _imp = f'{imp}.*'
                    modname, _ = parser._fetch_module_name(_imp, cat)
                    mod, _ = parser._fetch_module(_imp, cat)
                    tasks = {}
                    if mod is not None:
                        for name, obj in inspect.getmembers(mod, inspect.isclass):
                            if issubclass(obj, _task.Task) and obj.__module__ == modname:
                                tasks[name] = obj
                    if tasks:
                        for cls in tasks:
                            print(
                                indent(f'[green]{imp}.{cls}[/green]', _INDENT))
                            self._print_help(tasks[cls], 2)
