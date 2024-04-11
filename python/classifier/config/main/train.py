from __future__ import annotations

import logging
from datetime import datetime
from itertools import chain
from typing import TYPE_CHECKING

from classifier.task import ArgParser, EntryPoint, Model, converter

from ._utils import LoadTrainingSets, SelectDevice

if TYPE_CHECKING:
    from classifier.process.device import Device
    from classifier.task.model import ModelTrainer
    from torch.utils.data import StackDataset


class _train_model:
    def __init__(self, device: Device, datasets: StackDataset):
        self._device = device
        self._datasets = datasets

    def __call__(self, trainer: ModelTrainer):
        return trainer(self._device, self._datasets)


class Main(SelectDevice, LoadTrainingSets):
    argparser = ArgParser(
        prog="train",
        description="Train multiple models using one dataset.",
        workflow=[
            *LoadTrainingSets._workflow,
            ("main", "[blue]\[trainer, ...]=model.train()[/blue] initialize models"),
            ("sub", "[blue]trainer(device, datasets)[/blue] train models"),
        ],
    )
    argparser.add_argument(
        "--max-trainers",
        type=converter.int_pos,
        default=1,
        help="the maximum number of models to train in parallel",
    )

    def run(self, parser: EntryPoint):
        from concurrent.futures import ProcessPoolExecutor as Pool

        from classifier.process import status

        # load datasets in parallel
        datasets = self.load_training_sets(parser)
        # initialize datasets
        m_initializer: list[Model] = parser.mods["model"]
        timer = datetime.now()
        models = [*chain(*(m.train() for m in m_initializer))]
        logging.info(f"Initialized {len(models)} models in {datetime.now() - timer}")
        # train models in parallel
        with Pool(
            max_workers=self.opts.max_trainers,
            mp_context=status.context,
            initializer=status.initializer,
        ) as pool:
            results = [*pool.map(_train_model(self.device, datasets), models)]

        return {"models": results}
