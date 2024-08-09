from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import cached_property, reduce
from itertools import chain
from typing import TYPE_CHECKING

from base_class.utils import unique
from classifier.config.main._utils import progress_advance
from classifier.task import ArgParser, Dataset, converter, parse

if TYPE_CHECKING:
    import pandas as pd
    from base_class.root import Friend
    from classifier.df.io import FromRoot, ToTensor
    from classifier.df.tools import DFProcessor

# basic


class Dataframe(Dataset):
    def __init__(self):
        from classifier.df.io import ToTensor

        self._to_tensor = ToTensor()
        self._preprocessors: list[DFProcessor] = []
        self._postprocessors: list[DFProcessor] = []
        self._trainables: list[_load_df] = []

    @property
    def to_tensor(self):
        return self._to_tensor

    @property
    def preprocessors(self):
        return self._preprocessors

    @property
    def postprocessors(self):
        return self._postprocessors

    def train(self):
        for t in self._trainables:
            t.to_tensor = self.to_tensor
            t.postprocessors = self.postprocessors
        return self._trainables


class _load_df(ABC):
    to_tensor: ToTensor
    postprocessors: list[DFProcessor]

    def __call__(self):
        data = self.load()
        for p in self.postprocessors:
            data = p(data)
        return self.to_tensor.tensor(data)

    @abstractmethod
    def load(self) -> pd.DataFrame: ...


# ROOT


class LoadRoot(ABC, Dataframe):
    argparser = ArgParser()
    argparser.add_argument(
        "--files",
        action="extend",
        nargs="+",
        default=[],
        help="the paths to the ROOT files",
    )
    argparser.add_argument(
        "--filelists",
        action="extend",
        nargs="+",
        default=[],
        help=f"the paths to the filelists {parse.EMBED}",
    )
    argparser.add_argument(
        "--friends",
        action="extend",
        nargs="+",
        default=[],
        help="the paths to the json files with friend tree metadata",
    )
    argparser.add_argument(
        "--max-workers",
        type=converter.int_pos,
        default=1,
        help="the maximum number of workers to use when reading the ROOT files in parallel",
    )
    argparser.add_argument(
        "--chunksize",
        type=converter.int_pos,
        default=1_000_000,
        help="the size of chunk to read ROOT files",
    )
    argparser.add_argument(
        "--tree",
        default="Events",
        help="the name of the TTree",
    )

    def _parse_files(self, files: list[str], filelists: list[str]) -> list[str]:
        return unique(
            reduce(
                list.__add__,
                (parse.mapping(f, "file") or [] for f in filelists),
                files.copy(),
            )
        )

    def _parse_friends(self, friends: list[str]) -> list[Friend]:
        from base_class.root import Friend

        return [Friend.from_json(parse.mapping(f, "file")) for f in friends]

    def _from_root(self):
        yield self.from_root(), self.files

    def train(self):
        self._trainables.append(
            _load_df_from_root(
                *self._from_root(),
                max_workers=self.opts.max_workers,
                chunksize=self.opts.chunksize,
                tree=self.opts.tree,
            )
        )
        return super().train()

    @cached_property
    def files(self) -> list[str]:
        return self._parse_files(self.opts.files, self.opts.filelists)

    @cached_property
    def friends(self) -> list[Friend]:
        return self._parse_friends(self.opts.friends)

    @abstractmethod
    def from_root(self) -> FromRoot: ...


class LoadGroupedRoot(LoadRoot):
    argparser = ArgParser()
    argparser.add_argument(
        "--files",
        action="append",
        nargs="+",
        metavar=("GROUPS", "PATHS"),
        default=[],
        help="comma-separated groups and paths to the ROOT file",
    )
    argparser.add_argument(
        "--filelists",
        action="append",
        nargs="+",
        metavar=("GROUPS", "PATHS"),
        default=[],
        help=f"comma-separated groups and paths to the filelist {parse.EMBED}",
    )
    argparser.add_argument(
        "--friends",
        action="append",
        nargs="+",
        metavar=("GROUPS", "PATHS"),
        default=[],
        help="comma-separated groups and paths to the json file with the friend tree metadata",
    )

    def _from_root(self):
        files = self.files
        for k in files:
            yield self.from_root(k), files[k]

    @cached_property
    def files(self):
        files = parse.grouped_mappings(self.opts.files, ",")
        filelists = parse.grouped_mappings(self.opts.filelists, ",")
        return {
            k: self._parse_files(files.get(k, []), filelists.get(k, []))
            for k in set(files).union(filelists)
        }

    @cached_property
    def friends(self):
        return {
            k: self._parse_friends(v)
            for k, v in parse.grouped_mappings(self.opts.friends, ",").items()
        }

    @abstractmethod
    def from_root(self, groups: frozenset[str]) -> FromRoot: ...


class _fetch:
    def __init__(self, tree: str):
        self._tree = tree

    def __call__(self, path: str):
        from base_class.root import Chunk

        chunk = Chunk(source=path, name=self._tree, fetch=True)
        return chunk


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
        from classifier.monitor.progress import Progress
        from classifier.process import pool, status

        with ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=status.context,
            initializer=status.initializer,
        ) as executor:
            with Progress.new(
                total=sum(map(lambda x: len(x[1]), self._from_root)),
                msg=("files", "Fetching metadata"),
            ) as progress:
                chunks = [
                    list(
                        pool.submit(
                            executor,
                            _fetch(tree=self._tree),
                            files,
                            callbacks=[lambda _: progress_advance(progress)],
                        )
                    )
                    for _, files in self._from_root
                ]
            with Progress.new(
                total=sum(map(len, chain(*chunks))),
                msg=("events", "Loading"),
            ) as progress:
                dfs = [
                    *chain(
                        *(
                            pool.submit(
                                executor,
                                self._from_root[i][0].read,
                                Chunk.balance(
                                    self._chunksize,
                                    *chunks[i],
                                    common_branches=True,
                                ),
                                callbacks=[
                                    lambda x: progress_advance(progress, len(x))
                                ],
                            )
                            for i in range(len(chunks))
                        )
                    ),
                ]
        df = pd.concat(
            filter(lambda x: x is not None, dfs), ignore_index=True, copy=False
        )
        logging.info(
            "Loaded <DataFrame>:",
            f"entries: {len(df)}",
            f"columns: {sorted(df.columns)}",
        )
        return df
