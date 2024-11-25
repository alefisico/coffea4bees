from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING

from classifier.task import GlobalState

if TYPE_CHECKING:
    from base_class.system.eos import EOS, PathLike


FILE = ("python", "classifier", "config", "state", "__init__.py")


class RunInfo(GlobalState):
    main_task: str = None
    startup_time: datetime = datetime.now()
    singularity: bool = os.path.isdir("/.singularity.d")


class RepoInfo:
    user: str = "cms-cmu"
    repo: str = "coffea4bees"
    branch: str = "master"
    url: str = f"https://gitlab.cern.ch/{user}/{repo}/-/tree/{branch}/"

    _local: EOS = None

    @classmethod
    def get_url(cls, path: PathLike) -> str:
        from base_class.system.eos import EOS

        if cls._local is None:
            local = EOS(__file__)
            for i in range(len(FILE)):
                if local.name != FILE[-i - 1]:
                    i -= 1
                    break
                local = local.parent
            cls._local = local, cls.url + "".join(
                map(lambda x: x + "/", FILE[: -i - 1])
            )
        local, url = cls._local
        path = EOS(path)
        if not path.isin(local):
            return str(path)
        path = path.relative_to(local)
        return f"{url}{path}"


class MonitorInfo:
    backends: list[str] = [
        "console",
    ]
    components: list[str] = [
        "logging",
        "usage",
        "progress",
    ]
