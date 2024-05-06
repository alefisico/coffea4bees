from __future__ import annotations

import logging
from datetime import datetime
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING

from base_class.utils import unique
from classifier.task import ArgParser, EntryPoint, Main, converter
from classifier.utils import call

if TYPE_CHECKING:
    from classifier.task.dataset import Dataset


class SetupMultiprocessing(Main):
    argparser = ArgParser()
    argparser.add_argument(
        "--preload",
        action="extend",
        nargs="+",
        default=["torch"],
        help="preloaded imports when using multiprocessing",
    )

    def __init__(self):
        super().__init__()
        from classifier.process import get_context, status
        from classifier.process.initializer import torch_set_sharing_strategy

        status.context = get_context(
            method="forkserver", library="torch", preload=unique(self.opts.preload)
        )
        status.initializer.add(torch_set_sharing_strategy("file_system"))


class SelectDevice(Main):
    argparser = ArgParser()
    argparser.add_argument(
        "--device",
        nargs="+",
        default=["cuda"],
        help="the [green]torch.device[/green] used for training",
    )

    @cached_property
    def device(self):
        from classifier.process.device import Device

        return Device(*self.opts.device)


class LoadTrainingSets(SetupMultiprocessing):
    _workflow = [
        ("main", "[blue]\[loader, ...]=dataset.train()[/blue] initialize datasets"),
        ("sub", "[blue]loader()[/blue] load datasets"),
    ]
    argparser = ArgParser()
    argparser.add_argument(
        "--max-loaders",
        type=converter.int_pos,
        default=1,
        help="the maximum number of datasets to load in parallel",
    )

    def load_training_sets(self, parser: EntryPoint):
        from concurrent.futures import ProcessPoolExecutor as Pool

        from classifier.process import status
        from torch.utils.data import ConcatDataset, StackDataset

        # load datasets in parallel
        mods: list[Dataset] = parser.mods["dataset"]
        loaders = [*chain(*(k.train() for k in mods))]
        if len(loaders) == 0:
            raise ValueError("No dataset to load")
        logging.info(f"Loading {len(loaders)} datasets")
        timer = datetime.now()
        with Pool(
            max_workers=self.opts.max_loaders,
            mp_context=status.context,
            initializer=status.initializer,
        ) as pool:
            datasets = [*pool.map(call, loaders)]
        logging.info(f"Loaded {len(loaders)} datasets in {datetime.now() - timer}")
        # concatenate datasets
        keys = [set(d.keys()) for d in datasets]
        kept = set.intersection(*keys)
        ignored = set.union(*keys) - kept
        kept = sorted(kept)
        logging.info(f"The following keys will be kept: {kept}")
        if ignored:
            logging.warn(f"The following keys will be ignored: {sorted(ignored)}")
        datasets = {k: ConcatDataset(d[k] for d in datasets) for k in kept}
        logging.info(f"Loaded {len(next(iter(datasets.values())))} data entries")
        return StackDataset(**datasets)
