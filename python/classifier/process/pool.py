from concurrent.futures import Executor, Future
from queue import Queue
from typing import Callable, Generator, Generic, Iterable, Optional, TypeVar

_ResultT = TypeVar("_ResultT")


class _FuturePool(Generic[_ResultT]):
    def __init__(self, timeout: Optional[float], count: int):
        self._timeout = timeout
        self._count = count
        self._tasks = Queue()

    def __len__(self):
        return self._count

    def __iter__(self):
        for _ in range(self._count):
            yield self._tasks.get()
        self._count = 0
        self._tasks = None

    def _result(self, task: Future[_ResultT]):
        self._tasks.put(task.result(timeout=self._timeout))


def submit(
    e: Executor,
    fn: Callable[..., _ResultT],
    *args: Iterable,
    timeout: Optional[float] = None,
    callbacks: Iterable[Callable[[Future[_ResultT]], None]] = (),
) -> Generator[_ResultT, None, None]:
    tasks = [e.submit(fn, *arg) for arg in zip(*args)]
    results = _FuturePool(timeout, len(tasks))
    callbacks = [*callbacks, results._result]
    for task in tasks:
        for callback in callbacks:
            task.add_done_callback(callback)
    return results
