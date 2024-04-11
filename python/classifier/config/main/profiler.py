import logging
import os

from classifier.task import ArgParser, EntryPoint, Model, converter, parse

from .. import setting as cfg
from ._utils import LoadTrainingSets, SelectDevice
from .help import _print_mod


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

    def run(self, parser: EntryPoint):
        import numpy as np
        import torch
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
                torch.cuda.memory._record_memory_history()
                results[name] = trainer(self.device, datasets)
                torch.cuda.memory._dump_snapshot(
                    os.fspath(cfg.IO.profiler / f"snapshot_{name}.pkl")
                )
                torch.cuda.memory._record_memory_history(enabled=False)
        return {"models": results}
