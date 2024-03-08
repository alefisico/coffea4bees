import logging

from classifier.task import ArgParser, EntryPoint, main
from rich.console import Console
from rich.logging import RichHandler


class Main(main.Main):
    argparser = ArgParser(
        prog="debug",
        description="Debug tasks.",
        workflow=[
            ("main", f"call [blue]task.debug(logger)[/blue]"),
        ],
    )

    def __init__(self):
        super().__init__()
        self._console = Console(record=True)
        self._logger = logging.Logger("debug", logging.DEBUG)
        self._logger.addHandler(
            RichHandler(logging.DEBUG, console=self._console, markup=True)
        )

    def run(self, parser: EntryPoint):
        logger = self._logger
        for k, v in parser.mods.items():
            logger.debug(f"Checking [blue]{k}[/blue]...")
            args = parser.args[k]
            for t, arg in zip(v, args):
                logger.debug(
                    f'[green]{arg[0]}[/green] [yellow]{" ".join(arg[1])}[/yellow]'
                )
                try:
                    t.debug(logger)
                except NotImplementedError:
                    ...
                except Exception as e:
                    logger.error(f"{e}")
