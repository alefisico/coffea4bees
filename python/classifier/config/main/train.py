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


class _init_model:
    def __init__(self, dataset: StackDataset):
        self._dataset = dataset

    def __call__(self, model: Model):
        return model.train(self._dataset)


class _train_model:
    def __init__(self, device: Device):
        self._device = device

    def __call__(self, trainer: ModelTrainer):
        return trainer(self._device)


class Main(SelectDevice, LoadTrainingSets):
    argparser = ArgParser(
        prog='train',
        description='Train multiple models using one dataset.',
        workflow=[
            *LoadTrainingSets._workflow,
            ('sub', 'call [blue]model.train(dataset)[/blue]'),
            ('sub', 'train [blue]model[/blue]')
        ])
    argparser.add_argument(
        '--max-trainers', type=converter.int_pos, default=1, help='the maximum number of models to train in parallel')

    def run(self, parser: EntryPoint):
        from concurrent.futures import ProcessPoolExecutor as Pool

        from classifier.process import process_state

        # load datasets in parallel
        datasets = self.load_training_sets(parser)
        # TODO test below with new skimmed data
        # initialize datasets in parallel
        m_initializer = parser.mods['model']
        timer = datetime.now()
        with Pool(
            max_workers=self.opts.max_trainers,
            mp_context=process_state.context,
            initializer=process_state.initializer
        ) as pool:
            models = [*chain(*pool.map(_init_model(datasets), m_initializer))]
        logging.info(
            f'Initialized {len(models)} models in {datetime.now() - timer}')
        # train models in parallel
        with Pool(
            max_workers=self.opts.max_trainers,
            mp_context=process_state.context,
            initializer=process_state.initializer
        ) as pool:
            results = [*pool.map(_train_model(self.device), models)]

        return {'models': results}
