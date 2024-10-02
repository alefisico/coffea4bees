from classifier.task import ArgParser, EntryPoint, main
from classifier.task.special import WorkInProgress

from ._utils import SelectDevice


class Main(WorkInProgress, SelectDevice, main.Main):
    argparser = ArgParser(prog="evaluate")

    def run(self, parser: EntryPoint): ...  # TODO
