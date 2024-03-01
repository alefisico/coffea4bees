from classifier.task import ArgParser, EntryPoint

from ._utils import SelectDevice, SetupMultiprocessing


class Main(SelectDevice, SetupMultiprocessing):
    argparser = ArgParser(
        prog='evaluate',
        description='[red]Work in Progress[/red]')

    def run(self, parser: EntryPoint):
        ...  # TODO
