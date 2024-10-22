from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from base_class.root import Chain, Chunk, Friend

from ..monitor.progress import Progress, ProgressTracker
from ..process import status

if TYPE_CHECKING:
    from base_class.root.chain import NameMapping
    from base_class.system.eos import PathLike


@dataclass(kw_only=True)
class _merge_worker:
    chunk: Chunk = None
    chain: Chain
    name: str
    base_path: PathLike
    naming: str | NameMapping

    def new(self, chunk: Chunk):
        return replace(self, chunk=chunk)

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
        clean: bool = False,
    ):
        self._friends = friends
        self._job = _merge_worker(
            chain=Chain().add_friend(*friends, renaming=self._rename_column),
            name=friend_name,
            base_path=dump_base_path,
            naming=dump_naming,
        )
        self._step = step
        self._workers = workers
        self._clean = clean

    def __call__(self):
        # assume all friend trees have the same structure
        targets = [*self._friends[0].targets]
        result = _update_friend()
        with (
            Progress.new(
                total=sum(map(len, targets)),
                msg=("entries", "Merging", "k-folds"),
            ) as progress,
            ProcessPoolExecutor(
                max_workers=self._workers,
                mp_context=status.context,
                initializer=status.initializer,
            ) as pool,
        ):
            jobs = []
            for chunk in Chunk.balance(self._step, targets):
                job = pool.submit(self._job.new(chunk))
                job.add_done_callback(result)
                job.add_done_callback(_update_progress(progress, len(chunk)))
                jobs.append(job)
            wait(jobs)
            if self._clean:
                for friend in self._friends:
                    friend.reset(confirm=False, executor=pool)
        output = {"merged": result.friend}
        if not self._clean:
            output["original"] = [*self._friends]
        return output
