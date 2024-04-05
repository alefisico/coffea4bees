from __future__ import annotations

from typing import TYPE_CHECKING

from base_class.system.eos import EOS, PathLike
from classifier.task import GlobalState

if TYPE_CHECKING:
    from datetime import datetime


class RunInfo(GlobalState):
    main_task: str = None
    startup_time: datetime = None


class RepoInfo:
    user: str = "cms-cmu"
    repo: str = "coffea4bees"
    branch: str = "master"
    url: str = f"https://gitlab.cern.ch/{user}/{repo}/-/tree/{branch}/"

    _local: EOS = None

    @classmethod
    def get_url(cls, path: PathLike) -> str:
        if cls._local is None:
            cls._local = EOS(__file__)
            while cls._local.name != cls.repo:
                cls._local = cls._local.parent
        path = EOS(path)
        if not path.isin(cls._local):
            return str(path)
        path = path.relative_to(cls._local)
        return f"{cls.url}{path}"


class MonitorInfo:
    backends: tuple[str] = ("console",)
    components: tuple[str] = ("logging",)
