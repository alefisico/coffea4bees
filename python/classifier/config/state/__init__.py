import os
from datetime import datetime

from classifier.task import GlobalState
from classifier.task.state import _SET


class System(GlobalState):
    main_task: str = None
    startup_time: datetime = datetime.now()
    in_singularity: bool = os.path.isdir("/.singularity.d")


class Flags(GlobalState):
    test: bool = False
    debug: bool = False

    def _set(self, *names: str):
        for name in names:
            if name in self.__annotations__:
                value = getattr(self, name)
                setter = f"{_SET}{name}"
                if not value and hasattr(self, setter):
                    getattr(self, setter)()
                setattr(self, name, True)

    @classmethod
    def set(cls, *names: str):
        cls._set(cls, *names)

    @classmethod
    def set__debug(cls):
        from ..setting.monitor import Log

        Log.level = 10
