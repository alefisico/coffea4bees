from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.task import GlobalState

if TYPE_CHECKING:
    from datetime import datetime


class RunInfo(GlobalState):
    main_task: str = None
    startup_time: datetime = None
