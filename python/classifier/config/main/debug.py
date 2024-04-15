import logging

from classifier.task import ArgParser, EntryPoint, main
from classifier.task.special import InterfaceError

from .help import _print_mod


class Main(main.Main):
    argparser = ArgParser(
        prog="debug",
        description="Debug tasks.",
        workflow=[
            ("main", f"[blue]task.debug()[/blue] debug tasks"),
        ],
    )

    def __init__(self):
        super().__init__()

    def run(self, parser: EntryPoint):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        for k, v in parser.mods.items():
            logging.debug(f"Checking [blue]{k}[/blue]...")
            args = parser.args[k]
            for t, arg in zip(v, args):
                logging.debug(_print_mod(k, arg[0], arg[1], ""))
                try:
                    t.debug()
                except InterfaceError:
                    ...
                except Exception as e:
                    logging.error(e, exc_info=e)
