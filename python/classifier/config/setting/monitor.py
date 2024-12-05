from typing import Callable, Protocol, TypeVar

from base_class.utils.wrapper import MethodDecorator
from classifier.task.state import GlobalSetting

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
class Console(GlobalSetting):
    "Backend: console (rich)"

    enable: bool = True
    "enable the console backend"

    interval: float = 1.0
    "(seconds) interval to update the "
    fps: int = 10
    "frames per second"


class Web(GlobalSetting):  # TODO placeholder
    "Backend: web page (flask)"

    enable: bool = False
    "enable the web backend"


# functions
class Log(GlobalSetting):
    "Logging system"

    enable: bool = True
    "enable logging"
    file: str = "logs.html"
    "name of the log file"

    level: int = 20
    "logging level"
    forward_exception: bool = True
    "forward the uncaught exceptions to the monitor (set this to False or run a standalone monitor if some exceptions do not show up)"


class Progress(GlobalSetting):
    "Progress bars"

    enable: bool = True
    "enable progress bars"


class Usage(GlobalSetting):
    "Usage statistics"

    enable: bool = False
    "enable usage trackers (this will significantly slow down the program)"
    file: str = "usage.json"
    "name of the file to dump the raw usage data"

    interval: float = 1.0
    "(seconds) interval to update the usage"
    gpu: bool = True
    "track GPU usage"
    gpu_force_torch: bool = False
    "force to fetch GPU usage from pytorch instead of pynvml"


class Input(GlobalSetting):
    "Text input"

    enable: bool = True
    "enable text input"
