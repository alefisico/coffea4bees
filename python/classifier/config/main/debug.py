import logging

from classifier.task import ArgParser, EntryPoint, main


class Main(main.Main):
    argparser = ArgParser(
        prog="debug",
        description="Debug tasks.",
        workflow=[
            ("main", f"call [blue]task.debug()[/blue]"),
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
                logging.debug(
                    f'[green]{arg[0]}[/green] [yellow]{" ".join(arg[1])}[/yellow]'
                )
                try:
                    t.debug()
                except NotImplementedError:
                    ...
                except Exception as e:
                    logging.error(f"{e}")
