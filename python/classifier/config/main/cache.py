from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import TYPE_CHECKING

import fsspec
from classifier.task import ArgParser, EntryPoint

from ._utils import LoadTrainingSets

if TYPE_CHECKING:
    import numpy.typing as npt
    from base_class.system.eos import EOS
    from torch.utils.data import StackDataset


class Main(LoadTrainingSets):
    argparser = ArgParser(
        prog='cache',
        description='write the datasets to files, which can be loaded by [green]cache.Torch[/green]',
        workflow=[
            *LoadTrainingSets._workflow,
            ('sub', 'write chunks to disk'),
        ])
    argparser.add_argument(
        '--shuffle', action='store_true', help='shuffle the dataset before saving')
    argparser.add_argument(
        '--nchunks', type=int, help='number of chunks')
    argparser.add_argument(
        '--chunksize', type=int, help='size of each chunk, will be ignored if [yellow]--nchunks[/yellow] is given')
    argparser.add_argument(
        '--compression', choices=fsspec.available_compressions(), help='compression algorithm to use')
    argparser.add_argument(
        '--max-writers', type=int, default=1, help='the maximum number of files to write in parallel')

    def run(self, parser: EntryPoint):
        from concurrent.futures import ProcessPoolExecutor as Pool

        import numpy as np

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
        chunks = [chunks[i:i+chunksize] for i in range(0, size, chunksize)]

        timer = datetime.now()
        with Pool(
            max_workers=self.opts.max_writers,
            mp_context=self.mp_context,
            initializer=self.mp_initializer
        ) as pool:
            _ = pool.map(_save_cache(datasets, self.output, self.opts.compression),
                         zip(range(len(chunks)), chunks))
        logging.info(
            f'Wrote {size} entries to {len(chunks)} files in {datetime.now() - timer}')

        return {
            'size': size,
            'chunksize': chunksize,
            'shuffle': self.opts.shuffle,
            'compression': self.opts.compression,
        }


class _save_cache:
    def __init__(self, dataset: StackDataset, path: EOS, compression: str = None):
        self.dataset = dataset
        self.path = path
        self.compression = compression

    def __call__(self, args: tuple[int, npt.ArrayLike]):
        import torch
        from torch.utils.data import DataLoader, Subset

        from ..setting.default import Dataset as Setting

        chunk, indices = args
        subset = Subset(self.dataset, indices)
        chunks = [
            *DataLoader(subset, batch_size=Setting.dataloader_io_batch//len(self.dataset.datasets))]
        data = {
            k: torch.cat([c[k] for c in chunks])
            for k in self.dataset.datasets}
        with fsspec.open(self.path / f'chunk{chunk}.pt', 'wb', compression=self.compression) as f:
            torch.save(data, f)
