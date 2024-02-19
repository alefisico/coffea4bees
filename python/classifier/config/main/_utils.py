from __future__ import annotations

import logging
from datetime import datetime
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING

from classifier.task import task

if TYPE_CHECKING:
    from classifier.task.dataset import DatasetLoader, TrainableDataset
    from torch.utils.data import Dataset


class SetupMultiprocessing(task._Main):
    argparser = task.ArgParser()
    argparser.add_argument(
        '--mp-preload', action='extend', nargs='+', default=['torch'], help='preloaded imports when using multiprocessing')

    @cached_property
    def mp_context(self):
        from classifier import process

        return process.get_context(method='forkserver', library='torch', preload=self.opts.mp_preload)

    @cached_property
    def mp_initializer(self):
        from classifier.process.initializer import (
            MultiInitializer, inherit_context_initializer,
            torch_set_sharing_strategy)

        initializer = MultiInitializer(
            torch_set_sharing_strategy('file_system'))
        initializer.add(inherit_context_initializer(
            self.mp_context, initializer))
        return initializer


class SelectDevice(task._Main):
    argparser = task.ArgParser()
    argparser.add_argument(
        '--device', type=str, nargs='+', default=['cuda'], help='the [green]torch.device[/green] used for training')


class _load_datasets:
    def __init__(self):
        ...

    def __call__(self, loader: TrainableDataset):
        return loader()


class LoadDatasets(SetupMultiprocessing):
    argparser = task.ArgParser()
    argparser.add_argument(
        '--max-loader', type=int, default=1, help='the maximum number of dataset loaders to run in parallel')

    def load_datasets(self, parser: task.Parser) -> dict[str, Dataset]:
        from concurrent.futures import ProcessPoolExecutor as Pool

        from torch.utils.data import ConcatDataset

        # load datasets in parallel
        d_mods: list[DatasetLoader] = parser.mods['dataset']
        d_loaders = [*chain(*(k.train() for k in d_mods))]
        if len(d_loaders) == 0:
            raise ValueError('No dataset to load')
        timer = datetime.now()
        with Pool(
            max_workers=min(self.opts.max_loader, len(d_loaders)),
            mp_context=self.mp_context,
            initializer=self.mp_initializer
        ) as pool:
            datasets = [*pool.map(_load_datasets(), d_loaders)]
        logging.info(
            f'Loaded {len(d_loaders)} datasets in {datetime.now() - timer}')
        # concatenate datasets
        d_keys = [set(d.keys()) for d in datasets]
        kept = set.intersection(*d_keys)
        ignored = set.union(*d_keys) - kept
        kept = sorted(kept)
        logging.info(
            f'The following keys will be kept: {kept}')
        if ignored:
            logging.warning(
                f'The following keys will be ignored: {sorted(ignored)}')
        dataset = {k: ConcatDataset(d[k] for d in datasets) for k in kept}
        logging.info(
            f'Loaded {len(next(iter(dataset.values())))} data entries')
        return dataset
