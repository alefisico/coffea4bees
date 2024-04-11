from __future__ import annotations

import time
from dataclasses import dataclass

from rich.progress import BarColumn
from rich.progress import Progress as _Bar
from rich.progress import ProgressColumn, SpinnerColumn, TimeElapsedColumn

from ..config.setting import Monitor as cfg
from ..process.monitor import Proxy, Recorder, callback
from ..typetools import WithUUID
from ..utils import noop

_FORMAT = "%H:%M:%S"
_UNKNOWN = "+--:--:--"


@dataclass
class _ProgressTracker(WithUUID):
    msg: str
    total: int

    def __post_init__(self):
        self.start_t = time.time()
        self.updated_t = self.start_t
        self.source = Recorder.name()
        self._completed = 0
        self._step = None
        super().__init__()

    def _update(self, msg: str = None, updated: bool = False):
        if (msg is not None) and (msg != self.msg):
            self.msg = msg
            updated = True
        if updated:
            Progress._update(self)
        self._step = None

    def update(self, completed: int, msg: str = None):
        updated = completed > self._completed
        if updated:
            self.updated_t = time.time()
            self._completed = completed
            self._step = None
        self._update(msg, updated)

    def advance(self, step: int, msg: str = None):
        updated = step > 0
        if updated:
            self.updated_t = time.time()
            self._completed += step
            self._step = step
        self._update(msg, updated)

    @property
    def estimate(self):
        if self._completed > 0:
            return (
                self.updated_t - self.start_t
            ) / self._completed * self.total + self.start_t
        return None

    @property
    def is_finished(self):
        return self._completed >= self.total


class _EstimateColumn(ProgressColumn):
    def render(self, task):
        estimate = task.fields.get("estimate")
        text = _UNKNOWN
        if estimate is not None:
            diff = estimate - time.time()
            if diff > 0:
                text = f"+{time.strftime(_FORMAT, time.gmtime(diff))}"
        return text


class Progress(Proxy):
    _jobs: dict[tuple, _ProgressTracker]
    _console_ids: dict[tuple, str]
    _console_bar: _Bar

    def __init__(self):
        self._jobs = {}
        if cfg.console_enable:
            self._console_bar = _Bar(
                SpinnerColumn(),
                TimeElapsedColumn(),
                _EstimateColumn(),
                BarColumn(bar_width=None),
                "{task.completed}/{task.total} {task.description}",
                "\[{task.fields[source]}]",
                expand=True,
            )
            self._console_ids = {}

    @classmethod
    def new(cls, total: int, msg: str = "") -> _ProgressTracker:
        if cfg.progress_enable:
            job = _ProgressTracker(msg=msg, total=total)
            cls._update(job)
            return job
        return noop

    @callback(max_retry=1)
    def _update(self, new: _ProgressTracker):
        uuid = (new.source, new.uuid)
        old = self._jobs.get(uuid)
        if new._step is not None:
            if old is None:
                return
            new._completed = max(new._completed, old._completed + new._step)
        self._jobs[uuid] = new
        if new.is_finished:
            self._jobs.pop(uuid)

    @classmethod
    def _console_callback(cls):
        with cls.lock():
            jobs = cls._jobs.copy()

        for uuid, job in jobs.items():
            kwargs = {
                "description": job.msg,
                "completed": job._completed,
                "estimate": job.estimate,
                "source": Recorder.registered(job.source),
            }
            if uuid not in cls._console_ids:
                cls._console_ids[uuid] = cls._console_bar.add_task(
                    total=job.total, **kwargs
                )
            else:
                cls._console_bar.update(task_id=cls._console_ids[uuid], **kwargs)
        for uuid in set(cls._console_ids) - set(jobs):
            try:
                cls._console_bar.remove_task(cls._console_ids.pop(uuid))
            except KeyError:
                ...


def setup_monitor():
    if cfg.progress_enable:
        if cfg.console_enable:
            from .backends.console import Dashboard as _CD

            _CD.layout.add_row(Progress._console_bar)
            _CD.add(Progress._console_callback)
