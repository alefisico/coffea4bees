import time
from collections import defaultdict
from threading import Thread
from typing import TypedDict

import psutil

from ..config.setting import Monitor as cfg
from ..config.state import RunInfo
from ..process.monitor import Proxy, Recorder, callback

_MIB = 2**20
_CUDA = {"12.1": 254}  # MiB


class _Resource(TypedDict):
    time: float  # seconds
    cpu: float  # %
    memory: float  # MiB
    gpu: float  # MiB


class _Checkpoint(TypedDict):
    time: float  # seconds
    name: str


class Usage(Proxy):
    _records_local: list[_Resource] = []
    _tracker: Thread = None
    _head: int = 0

    # GPU
    _n_gpu: int = None
    _torch_calibration: int = None  # MiB
    _pynvml_handles: list = None
    _pynvml_unavailable = RunInfo.singularity

    def __init__(self):
        self._records: dict[str, list[_Resource]] = defaultdict(list)
        self._checkpoints: dict[str, list[_Checkpoint]] = defaultdict(list)

    @classmethod
    def start(cls):
        cls._tracker = Thread(target=cls._track, daemon=True)
        cls._tracker.start()

    @classmethod
    def checkpoint(cls, name: str):
        if cfg.usage_enable:
            checkpoint = {"time": time.time(), "name": name}
            end = len(cls._records_local)
            records = cls._records_local[cls._head : end]
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
            cpu = process.cpu_percent(cfg.usage_update_interval)
            mem = process.memory_info().rss / _MIB
            for child in process.children(recursive=True):
                cpu += child.cpu_percent()
                mem += child.memory_info().rss / _MIB
                pids.append(child.pid)
            # GPU
            if cfg.usage_gpu:
                if cfg.usage_gpu_force_torch or cls._pynvml_unavailable:
                    gpu = cls._gpu_torch()
                else:
                    gpu = cls._gpu()
                    if gpu is None:
                        cls._pynvml_unavailable = True
                        gpu = cls._gpu_torch()
                    else:
                        gpu = sum(gpu.get(pid, 0) for pid in pids)
            else:
                gpu = 0
            cls._records_local.append(
                {"time": now, "cpu": cpu, "memory": mem, "gpu": float(gpu)}
            )
            time.sleep(cfg.usage_update_interval)

    @classmethod
    def _gpu(cls):
        import pynvml

        if cls._n_gpu is None:
            try:
                pynvml.nvmlInit()
                cls._n_gpu = pynvml.nvmlDeviceGetCount()
            except pynvml.NVMLError:
                return None
        if cls._n_gpu > 0:
            if cls._pynvml_handles is None:
                cls._pynvml_handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(cls._n_gpu)
                ]
            gpu = defaultdict(float)
            for handle in cls._pynvml_handles:
                for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    gpu[p.pid] += (p.usedGpuMemory or 0) / _MIB
            return gpu
        return {}

    @classmethod
    def _gpu_torch(cls):
        import torch

        if cls._n_gpu is None:
            cls._n_gpu = torch.cuda.device_count()
        if cls._n_gpu > 0:
            if cls._torch_calibration is None:
                cls._torch_calibration = _CUDA.get(torch.version.cuda, 0)
            gpu = 0
            for i in range(cls._n_gpu):
                reserved = torch.cuda.memory_reserved(i)
                if reserved > 0:
                    gpu += (reserved / _MIB) + cls._torch_calibration
            return gpu
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
    if cfg.usage_enable:
        Usage.start()


def setup_monitor():
    if cfg.usage_enable:
        Usage.start()
        Recorder.to_dump(cfg.file_usage, Usage.serialize)
