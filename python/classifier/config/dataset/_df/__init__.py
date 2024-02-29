from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property, reduce
from itertools import chain
from typing import TYPE_CHECKING, Callable

from base_class.utils import unique
from classifier.task import ArgParser, Dataset, converter, parsers

if TYPE_CHECKING:
    import pandas as pd
    from base_class.root import Friend
    from classifier.df.io import FromRoot, ToTensor

# basic


class Dataframe(Dataset):
    def __init__(self):
        from classifier.df.io import ToTensor

        self._to_tensor = ToTensor()
        self._preprocessors: list[Callable[[pd.DataFrame], pd.DataFrame]] = []
        self._postprocessors: list[Callable[[pd.DataFrame], pd.DataFrame]] = []
        self._trainables: list[_load_df] = []

    @property
    def to_tensor(self):
        return self._to_tensor

    @property
    def preprocessors(self):
        return self._preprocessors

    def train(self):
        for t in self._trainables:
            t.to_tensor = self.to_tensor
            t.postprocessors = self._postprocessors
        return self._trainables


class _load_df(ABC):
    to_tensor: ToTensor
    postprocessors: list[Callable[[pd.DataFrame], pd.DataFrame]]

    def __call__(self):
        data = self.load()
        for p in self.postprocessors:
            data = p(data)
        return self.to_tensor.tensor(data)

    @abstractmethod
    def load(self) -> pd.DataFrame:
        ...

# ROOT


class LoadRoot(ABC, Dataframe):
    argparser = ArgParser()
    argparser.add_argument(
        '--files', action='extend', nargs='+', default=[], help='the paths to the ROOT files')
    argparser.add_argument(
        '--filelists', action='extend', nargs='+', default=[], help='the paths to the filelists')
    argparser.add_argument(
        '--friends', action='extend', nargs='+', default=[], help='the paths to the json files with friend tree metadata')
    argparser.add_argument(
        '--max-workers', type=converter.int_pos, default=1, help='the maximum number of workers to use when reading the ROOT files in parallel')
    argparser.add_argument(
        '--chunksize', type=converter.int_pos, default=1_000_000, help='the size of chunk to read ROOT files')
    argparser.add_argument(
        '--tree', default='Events', help='the name of the TTree')

    def _parse_files(self, files: list[str], filelists: list[str]) -> list[str]:
        return unique(reduce(
            list.__add__,
            (parsers.parse_dict(f'file:///{f}') for f in filelists),
            files.copy()
        ))

    def _parse_friends(self, friends: list[str]) -> list[Friend]:
        from base_class.root import Friend
        return [Friend.from_json(parsers.parse_dict(f'file:///{f}')) for f in friends]

    def _from_root(self):
        yield self.from_root, self.files

    def train(self):
        self._trainables.append(
            _load_df_from_root(
                *self._from_root(),
                max_workers=self.opts.max_workers,
                chunksize=self.opts.chunksize,
                tree=self.opts.tree
            ))
        return super().train()

    @cached_property
    def files(self) -> list[str]:
        return self._parse_files(self.opts.files, self.opts.filelists)

    @cached_property
    def friends(self) -> list[Friend]:
        return self._parse_friends(self.opts.friends)

    @property
    @abstractmethod
    def from_root(self) -> FromRoot:
        ...


class LoadGroupedRoot(LoadRoot):
    argparser = ArgParser()
    argparser.add_argument(
        '--files', action='append', nargs=2, metavar=('GROUP', 'PATH'), default=[], help='the group and path to the ROOT file')
    argparser.add_argument(
        '--filelists', action='append', nargs=2, metavar=('GROUP', 'PATH'), default=[], help='the group and path to the filelist')
    argparser.add_argument(
        '--friends', action='append', nargs=2, metavar=('GROUP', 'PATH'), default=[], help='the group and path to the json file with the friend tree metadata')

    def _from_root(self):
        from_root = self.from_root
        files = self.files
        for k in set(from_root).intersection(files):
            yield from_root[k], files[k]

    @cached_property
    def files(self) -> dict[str, list[str]]:
        files = parsers.parse_group(self.opts.files)
        filelists = parsers.parse_group(self.opts.filelists)
        return {
            k: self._parse_files(files.get(k, []), filelists.get(k, []))
            for k in set(files).union(filelists)}

    @cached_property
    def friends(self):
        return {
            k: self._parse_friends(v)
            for k, v in parsers.parse_group(self.opts.friends).items()}

    @property
    @abstractmethod
    def from_root(self) -> dict[str, FromRoot]:
        ...


class _load_df_from_root(_load_df):
    def __init__(
        self,
        *from_root: tuple[FromRoot, list[str]],
        max_workers: int,
        chunksize: int,
        tree: str,
    ):
        self._from_root = from_root
        self._max_workers = max_workers
        self._chunksize = chunksize
        self._tree = tree

    def load(self) -> pd.DataFrame:
        from concurrent.futures import ProcessPoolExecutor

        import pandas as pd
        from base_class.root import Chunk
        from classifier.process import default

        dfs = []
        with ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=default.context,
            initializer=default.initializer,
        ) as pool:
            chunks = []
            for _, files in self._from_root:
                chunks.append(
                    pool.map(Chunk._fetch, (Chunk(f, self._tree) for f in files)))
            for i in range(len(chunks)):
                balanced = Chunk.balance(
                    self._chunksize, *chunks[i], common_branches=True)
                dfs.append(pool.map(self._from_root[i][0].read, balanced))
        return pd.concat(chain(*dfs), ignore_index=True)
