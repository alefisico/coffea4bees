import time
from collections import defaultdict
from subprocess import check_output
from threading import Thread
from typing import TypedDict

import psutil

from ..config.setting import Monitor as Setting
from ..process.monitor import Proxy, Recorder, callback

_NVIDIA_SMI = "nvidia-smi"
_MIB = 2**20


class _Resource(TypedDict):
    time: float  # seconds
    cpu: float  # %
    memory: float  # MiB
    gpu: float  # MiB


class _Checkpoint(TypedDict):
    time: float  # seconds
    name: str


class Usage(Proxy):
    _records: list[_Resource] = []
    _has_gpu: bool = None
    _tracker: Thread = None
    _head: int = 0

    def __init__(self):
        self._data: dict[str, list[_Resource]] = defaultdict(list)
        self._checkpoint: dict[str, list[_Checkpoint]] = defaultdict(list)

    @classmethod
    def start(cls):
        cls._tracker = Thread(target=cls._track, daemon=True)
        cls._tracker.start()

    @classmethod
    def checkpoint(cls, name: str):
        if Setting.track_usage:
            checkpoint = {"time": time.time(), "name": name}
            end = len(cls._records)
            records = cls._records[cls._head : end]
            cls._head = end
            cls.send(Recorder.name(), checkpoint, records)

    @callback
    def send(self, process: str, checkpoint: _Checkpoint, records: list[_Resource]):
        process = Recorder.registered(process)
        self._data[process].extend(records)
        self._checkpoint[process].append(checkpoint)

    @classmethod
    def _track(cls):
        while True:
            now = time.time()
            process = psutil.Process()
            gpu_mems = cls._gpu()
            cpu = process.cpu_percent()
            mem = process.memory_info().rss / _MIB
            gpu = gpu_mems.get(process.pid, 0)
            for child in process.children(recursive=True):
                cpu += child.cpu_percent()
                mem += child.memory_info().rss / _MIB
                gpu += gpu_mems.get(child.pid, 0)
            cls._records.append({"time": now, "cpu": cpu, "memory": mem, "gpu": gpu})
            time.sleep(Setting.track_usage_interval)

    @classmethod
    def _gpu(cls):
        if cls._has_gpu is None:
            try:
                check_output([_NVIDIA_SMI])
                cls._has_gpu = True
            except:
                cls._has_gpu = False
        if cls._has_gpu:
            mem = check_output(
                [
                    _NVIDIA_SMI,
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ]
            ).decode()
            return dict(map(int, line.split(", ")) for line in mem.splitlines())
        return {}

    @classmethod
    def serialize(cls):
        import json

        return json.dumps(
            {
                "checkpoint": cls._checkpoint,
                "data": cls._data,
            }
        ).encode()


def setup_reporter():
    if Setting.track_usage:
        Usage.start()


def setup_monitor():
    if Setting.track_usage:
        Usage.start()
        Recorder.to_dump(Setting.file_usage, Usage.serialize)
