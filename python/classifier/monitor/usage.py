import time
from collections import defaultdict
from subprocess import check_output
from threading import Thread
from typing import TypedDict

import psutil

from ..config.setting import Monitor as Setting
from ..config.state import RunInfo
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
    _locals: list[_Resource] = []
    _has_gpu: bool = None
    _tracker: Thread = None
    _head: int = 0

    def __init__(self):
        self._records: dict[str, list[_Resource]] = defaultdict(list)
        self._checkpoints: dict[str, list[_Checkpoint]] = defaultdict(list)

    @classmethod
    def start(cls):
        cls._tracker = Thread(target=cls._track, daemon=True)
        cls._tracker.start()

    @classmethod
    def checkpoint(cls, name: str):
        if Setting.track_usage:
            checkpoint = {"time": time.time(), "name": name}
            end = len(cls._locals)
            records = cls._locals[cls._head : end]
            cls._head = end
            cls._checkpoint(Recorder.name(), checkpoint, records)

    @callback
    def _checkpoint(
        self, process: str, checkpoint: _Checkpoint, records: list[_Resource]
    ):
        process = Recorder.registered(process)
        self._records[process].extend(records)
        self._checkpoints[process].append(checkpoint)

    @classmethod
    def _track(cls):
        while True:
            now = time.time()
            process = psutil.Process()
            pids = [process.pid]
            # CPU, memory
            cpu = process.cpu_percent(Setting.usage_update_interval)
            mem = process.memory_info().rss / _MIB
            for child in process.children(recursive=True):
                cpu += child.cpu_percent()
                mem += child.memory_info().rss / _MIB
                pids.append(child.pid)
            # GPU
            if Setting.usage_gpu:
                if RunInfo.singularity:
                    gpu = cls._gpu_singularity()
                else:
                    gpu = cls._gpu()
                    gpu = sum(gpu.get(pid, 0) for pid in pids)
            else:
                gpu = 0
            cls._locals.append({"time": now, "cpu": cpu, "memory": mem, "gpu": gpu})
            time.sleep(Setting.usage_update_interval)

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
    def _gpu_singularity(cls):
        import torch

        if cls._has_gpu is None:
            cls._has_gpu = torch.cuda.is_available()
        if cls._has_gpu:
            return (
                sum(
                    torch.cuda.memory_reserved(i)
                    for i in range(torch.cuda.device_count())  # TODO calibration
                )
                / _MIB
            )
        return 0

    @classmethod
    def serialize(cls):
        import json

        return json.dumps(
            {
                "checkpoints": cls._checkpoints,
                "records": cls._records,
            }
        ).encode()


def setup_reporter():
    if Setting.track_usage:
        Usage.start()


def setup_monitor():
    if Setting.track_usage:
        Usage.start()
        Recorder.to_dump(Setting.file_usage, Usage.serialize)
