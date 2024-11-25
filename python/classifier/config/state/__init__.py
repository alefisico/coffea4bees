from __future__ import annotations

import os
from datetime import datetime

from classifier.task import GlobalState


class RunInfo(GlobalState):
    main_task: str = None
    startup_time: datetime = datetime.now()
    in_singularity: bool = os.path.isdir("/.singularity.d")
