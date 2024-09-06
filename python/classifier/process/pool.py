from concurrent.futures import Executor, Future
from queue import Queue
from typing import Callable, Generator, Generic, Iterable, Optional, ParamSpec, TypeVar

_SubmitT = TypeVar("_SubmitT")
_SubmitP = ParamSpec("_SubmitP")


class _FuturePool(Generic[_SubmitT]):
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

    def _result(self, task: Future[_SubmitT]):
        self._tasks.put(task.result(timeout=self._timeout))


def submit(
    e: Executor,
    fn: Callable[_SubmitP, _SubmitT],
    *args: Iterable,
    callbacks: Iterable[Callable[_SubmitP, Callable[[Future[_SubmitT]], None]]] = (),
    timeout: Optional[float] = None,
) -> Generator[_SubmitT, None, None]:
    args = [*zip(*args)]
    results = _FuturePool(timeout, len(args))
    callbacks = [*callbacks, lambda *_: results._result]
    for arg in args:
        task = e.submit(fn, *arg)
        for callback in callbacks:
            task.add_done_callback(callback(*arg))
    return results
