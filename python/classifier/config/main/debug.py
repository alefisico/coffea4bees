import logging

from classifier.task import ArgParser, EntryPoint, main


class Main(main.Main):
    argparser = ArgParser(
        prog='debug',
        description='[red]Work in Progress[/red] Debug selected tasks.')

    def run(self, parser: EntryPoint):
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug('Initialization complete. Exiting...')
