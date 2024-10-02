from itertools import chain

from classifier.task import Analysis, ArgParser, EntryPoint, main
from classifier.utils import call

from .. import setting as cfg
from ._utils import progress_advance


class Main(main.Main):
    _no_state = True

    argparser = ArgParser(
        prog="analyze",
        description="Run standalone analysis.",
        workflow=[
            ("main", "[blue]\[analyzer, ...]=analysis.analyze()[/blue] initialize"),
            ("sub", "[blue]analyzer()[/blue] run"),
        ],
    )

    @classmethod
    def prelude(cls):
        cfg.IO.monitor = None

    def run(self, _): ...


def run_analyzer(parser: EntryPoint, output: dict):
    from concurrent.futures import ProcessPoolExecutor

    from classifier.monitor import Index
    from classifier.monitor.progress import Progress
    from classifier.process import pool, status

    analysis: list[Analysis] = parser.mods["analysis"]
    analyzers = [*chain(*(a.analyze(output) for a in analysis))]
    if not analyzers:
        return []

    with (
        ProcessPoolExecutor(
            max_workers=cfg.Analysis.max_workers,
            mp_context=status.context,
            initializer=status.initializer,
        ) as executor,
        Progress.new(total=len(analyzers), msg=("analysis", "Running")) as progress,
    ):
        results = [
            *pool.submit(
                executor,
                call,
                analyzers,
                callbacks=[lambda _: progress_advance(progress)],
            )
        ]
    Index.render()
    return [*filter(lambda x: x is not None, results)]
