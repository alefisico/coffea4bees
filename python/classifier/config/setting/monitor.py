from typing import Callable, Protocol, TypeVar

from classifier.task.state import Cascade

from . import Monitor


class MonitorComponentConfig(Protocol):
    enable: bool
    file: str


class _monitor_status_checker:
    def __init__(
        self,
        func,
        dependencies: tuple[MonitorComponentConfig],
        default=None,
        is_callable=False,
    ):
        self.func = func
        self.cfgs = dependencies + (Monitor,)
        self.default = default
        self.is_callable = is_callable

    def __call__(self, *args, **kwargs):
        if all(cfg.enable for cfg in self.cfgs):
            return self.func(*args, **kwargs)
        if self.is_callable:
            return self.default()
        else:
            return self.default


_FuncT = TypeVar("_FuncT")


def check(
    *dependencies: MonitorComponentConfig, default=None, is_callable=False
) -> Callable[[_FuncT], _FuncT]:
    return lambda func: _monitor_status_checker(
        func, dependencies, default, is_callable
    )


# backends
class Console(Cascade):
    enable: bool = True
    interval: float = 1.0  # seconds
    fps: int = 10


class Web(Cascade):  # TODO placeholder
    enable: bool = False


# functions
class Log(Cascade):
    enable: bool = True
    file: str = "logs.html"

    level: int = 20


class Progress(Cascade):
    enable: bool = True


class Usage(Cascade):
    enable: bool = True
    file: str = "usage.json"

    interval: float = 1.0  # seconds
    gpu: bool = True
    gpu_force_torch: bool = False
