import logging
import os

from classifier.task import ArgParser, EntryPoint, Model, converter, parse

from .. import setting as cfg
from ._utils import LoadTrainingSets, SelectDevice
from .help import _print_mod

_PROFILE_DEFAULT = {
    "record_shapes": True,
    "profile_memory": True,
    "with_stack": True,
    "with_flops": True,
    "with_modules": True,
}
_PROFILE_SCHEDULE_DEFAULT = {
    "wait": 0,
    "warmup": 0,
    "active": 1,
}


class Main(SelectDevice, LoadTrainingSets):
    argparser = ArgParser(
        prog="profiler",
        description="""Run torch profiler on models.
[1] https://pytorch.org/blog/understanding-gpu-memory-1/
[2] https://pytorch.org/blog/understanding-gpu-memory-2/""",
        workflow=[
            *LoadTrainingSets._workflow,
            ("main", "call [blue]model.profile()[/blue]"),
            ("main", "train [blue]model[/blue] with profiler enabled"),
        ],
    )
    argparser.add_argument(
        "--dataset-size",
        type=converter.int_pos,
        default=100,
        help=f"size of dataset",
    )
    argparser.add_argument(
        "--profile-options",
        type=parse.mapping,
        default="",
        help=f"profiling options {parse.EMBED}",
    )
    argparser.add_argument(
        "--profile-schedule",
        type=parse.mapping,
        default="",
        help=f"profiling schedule {parse.EMBED}",
    )
    argparser.add_argument(
        "--profile-activities",
        nargs="+",
        default=["CPU", "CUDA"],
        help="profiling activities",
    )

    def run(self, parser: EntryPoint):
        import numpy as np
        from torch.profiler import ProfilerActivity, profile, schedule
        from torch.utils.data import Subset

        # load datasets in parallel
        dataset = self.load_training_sets(parser)
        datasets = Subset(
            dataset, np.random.choice(len(dataset), size=self.opts.dataset_size)
        )
        # initialize datasets
        m_initializer: list[Model] = parser.mods["model"]
        args = parser.args["model"]
        results = {}
        # train models in sequence
        for i, model in enumerate(m_initializer):
            logging.info(f"Initalizing {_print_mod('model', *args[i], ' ')}")
            trainers = model.train()
            for j, trainer in enumerate(trainers):
                name = f"{i}.{j}"
                with profile(
                    activities=[
                        getattr(ProfilerActivity, a)
                        for a in self.opts.profile_activities
                    ],
                    schedule=schedule(
                        **(_PROFILE_SCHEDULE_DEFAULT | self.opts.profile_schedule)
                    ),
                    **(_PROFILE_DEFAULT | self.opts.profile_options),
                ) as p:
                    results[name] = trainer(self.device, datasets)
                p.export_memory_timeline(
                    os.fspath(cfg.IO.profiler / f"profiler_{name}.html")
                )
        return {"models": results}
