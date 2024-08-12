from typing import Callable, Protocol, TypeVar

from base_class.utils.wrapper import MethodDecorator
from classifier.task.state import Cascade

from . import Monitor


class MonitorComponentConfig(Protocol):
    enable: bool


class MonitorComponentStatus(MethodDecorator):
    def __init__(
        self,
        func: Callable,
        dependencies: tuple[MonitorComponentConfig],
        default=None,
        is_callable: bool = False,
    ):
        super().__init__(func)
        self._cfgs = dependencies + (Monitor,)
        self._default = default
        self._is_callable = is_callable

    def __call__(self, *args, **kwargs):
        if all(cfg.enable for cfg in self._cfgs):
            return self._func(*args, **kwargs)
        if self._is_callable:
            return self._default()
        else:
            return self._default


_FuncT = TypeVar("_FuncT")


def check(
    *dependencies: MonitorComponentConfig,
    default=None,
    is_callable: bool = False,
) -> Callable[[_FuncT], _FuncT]:
    return lambda func: MonitorComponentStatus(func, dependencies, default, is_callable)


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
    enable: bool = False
    file: str = "usage.json"

    interval: float = 1.0  # seconds
    gpu: bool = True
    gpu_force_torch: bool = False
