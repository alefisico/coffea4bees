from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from base_class.root import Chain, Chunk, Friend

from ..monitor.progress import Progress, ProgressTracker
from ..process import status
from .dataset import _chunk_processor

if TYPE_CHECKING:
    from base_class.root.chain import NameMapping
    from base_class.system.eos import PathLike


@dataclass(kw_only=True)
class _merge_worker(_chunk_processor):
    chain: Chain
    name: str
    base_path: PathLike
    naming: str | NameMapping

    def __call__(self) -> Friend:
        chain = self.chain.copy().add_chunk(self.chunk)
        data = chain.concat(library="pd", friend_only=True)
        data = {
            k: np.nanmean(data.loc[:, k], axis=1)
            for k in data.columns.get_level_values(0)
        }
        with Friend(name=self.name).auto_dump(
            base_path=self.base_path, naming=self.naming
        ) as friend:
            friend.add(self.chunk, data)
            return friend


@dataclass
class _update_friend:
    friend: Friend = None

    def __call__(self, friend: Future[Friend]):
        friend = friend.result()
        if self.friend is None:
            self.friend = friend
        else:
            self.friend += friend


@dataclass
class _update_progress:
    progress: ProgressTracker
    step: int

    def __call__(self, _):
        self.progress.advance(self.step)


class merge_kfolds:
    @staticmethod
    def _rename_column(friend: str, branch: str):
        return (branch, friend)

    def __init__(
        self,
        *friends: Friend,
        step: int,
        workers: int,
        friend_name: str,
        dump_base_path: PathLike,
        dump_naming: str | NameMapping = ...,
    ):
        # assume all friend trees have the same structure
        self._targets = [*friends[0].targets]
        self._job = _merge_worker(
            chain=Chain().add_friend(*friends, renaming=self._rename_column),
            name=friend_name,
            base_path=dump_base_path,
            naming=dump_naming,
        )
        self._step = step
        self._workers = workers

    def __call__(self):
        result = _update_friend()
        with (
            Progress.new(
                total=sum(map(len, self._targets)),
                msg=("entries", "Merging", "k-folds"),
            ) as progress,
            ProcessPoolExecutor(
                max_workers=self._workers,
                mp_context=status.context,
                initializer=status.initializer,
            ) as pool,
        ):
            for chunk in Chunk.balance(self._step, *self._targets):
                job = pool.submit(self._job.new(chunk))
                job.add_done_callback(result)
                job.add_done_callback(_update_progress(progress, len(chunk)))
        return result.friend
