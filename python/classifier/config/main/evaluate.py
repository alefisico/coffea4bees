from classifier.task import ArgParser, EntryPoint
from classifier.task.special import WorkInProgress

from ._utils import SelectDevice, SetupMultiprocessing


class Main(WorkInProgress, SelectDevice, SetupMultiprocessing):
    argparser = ArgParser(prog="evaluate")

    def run(self, parser: EntryPoint): ...  # TODO
