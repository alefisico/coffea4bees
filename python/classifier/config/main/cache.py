from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import TYPE_CHECKING

import fsspec
from classifier.nn.dataset import io_loader
from classifier.task import ArgParser, EntryPoint, converter

from ..setting.default import IO as IOSetting
from ._utils import LoadTrainingSets

if TYPE_CHECKING:
    import numpy.typing as npt
    from base_class.system.eos import EOS
    from torch.utils.data import StackDataset


class Main(LoadTrainingSets):
    argparser = ArgParser(
        prog="cache",
        description="Save datasets to disk. Use [blue]--dataset[/blue] [green]cache.Torch[/green] to load.",
        workflow=[
            *LoadTrainingSets._workflow,
            ("sub", "write chunks to disk"),
        ],
    )
    argparser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle the dataset before saving",
    )
    argparser.add_argument(
        "--nchunks",
        type=converter.int_pos,
        help="number of chunks",
    )
    argparser.add_argument(
        "--chunksize",
        type=converter.int_pos,
        help="size of each chunk, will be ignored if [yellow]--nchunks[/yellow] is given",
    )
    argparser.add_argument(
        "--compression",
        choices=fsspec.available_compressions(),
        help="compression algorithm to use",
    )
    argparser.add_argument(
        "--max-writers",
        type=converter.int_pos,
        default=1,
        help="the maximum number of files to write in parallel",
    )
    argparser.remove_argument("--save-state")
    defaults = {"save_state": True}

    def run(self, parser: EntryPoint):
        from concurrent.futures import ProcessPoolExecutor as Pool

        import numpy as np
        from classifier.process import status

        datasets = self.load_training_sets(parser)
        size = len(datasets)
        chunks = np.arange(size)
        if self.opts.shuffle:
            np.random.shuffle(chunks)
        if self.opts.nchunks is not None:
            chunksize = math.ceil(size / self.opts.nchunks)
        elif self.opts.chunksize is not None:
            chunksize = self.opts.chunksize
        else:
            chunksize = size
        chunks = [chunks[i : i + chunksize] for i in range(0, size, chunksize)]

        timer = datetime.now()
        with Pool(
            max_workers=self.opts.max_writers,
            mp_context=status.context,
            initializer=status.initializer,
        ) as pool:
            _ = pool.map(
                _save_cache(datasets, IOSetting.output, self.opts.compression),
                zip(range(len(chunks)), chunks),
            )
        logging.info(
            f"Wrote {size} entries to {len(chunks)} files in {datetime.now() - timer}"
        )

        return {
            "size": size,
            "chunksize": chunksize,
            "shuffle": self.opts.shuffle,
            "compression": self.opts.compression,
        }


class _save_cache:
    def __init__(self, dataset: StackDataset, path: EOS, compression: str = None):
        self.dataset = dataset
        self.path = path
        self.compression = compression

    def __call__(self, args: tuple[int, npt.ArrayLike]):
        import torch
        from torch.utils.data import Subset

        chunk, indices = args
        subset = Subset(self.dataset, indices)
        chunks = [*io_loader(subset)]
        data = {k: torch.cat([c[k] for c in chunks]) for k in self.dataset.datasets}
        with fsspec.open(
            self.path / f"chunk{chunk}.pt", "wb", compression=self.compression
        ) as f:
            torch.save(data, f)
