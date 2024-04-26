import time
from collections import defaultdict
from threading import Thread
from typing import TypedDict

import psutil

from ...config.setting import Monitor as cfg
from ...config.state import RunInfo
from ...process.monitor import Node, Proxy, Recorder, callback

_MIB = 2**20
_CUDA = {"12.1": 254}  # MiB


class Resource(TypedDict):
    time: int  # ns
    cpu: dict[int, float]  # pid: cpu %
    memory: dict[int, float]  # pid: memory MiB
    gpu: dict[int, float]  # pid: gpu memory MiB


class Checkpoint(TypedDict):
    time: float  # s
    name: tuple[str, ...]
    pids: list[int]


class Usage(Proxy):
    _records_local: list[Resource] = []
    _tracker: Thread = None
    _head: int = 0

    # GPU
    _n_gpu: int = None
    _torch_calibration: int = None  # MiB
    _pynvml_handles: list = None
    _pynvml_unavailable = RunInfo.singularity

    # state
    _running = False

    def __init__(self):
        self._records: dict[Node, list[Resource]] = defaultdict(list)
        self._checkpoints: dict[Node, list[Checkpoint]] = defaultdict(list)

    @classmethod
    def start(cls):
        if cfg.usage_enable:
            cls._running = True
            cls._tracker = Thread(target=cls._track, daemon=True)
            cls._tracker.start()

    @classmethod
    def stop(cls):
        if cls._tracker is not None:
            cls._running = False
            cls._tracker.join()
            cls._tracker = None
            cls._records_local = []

    @classmethod
    def checkpoint(cls, *tags: str):
        if cfg.usage_enable and cls._running:
            start_t = time.time_ns()
            end = len(cls._records_local)
            records = cls._records_local[cls._head : end]
            cls._head = end
            end_t = time.time_ns()
            cls._checkpoint(
                Recorder.node(), {"time": (start_t + end_t) // 2, "name": tags}, records
            )

    @callback
    def _checkpoint(self, node: Node, checkpoint: Checkpoint, records: list[Resource]):
        self._records[node].extend(records)
        self._checkpoints[node].append(checkpoint)

    @classmethod
    def _track(cls):
        while cls._running:
            start_t = time.time_ns()
            p = psutil.Process()
            # CPU, memory
            cpu = {p.pid: p.cpu_percent(cfg.usage_update_interval)}
            mem = {p.pid: p.memory_info().rss / _MIB}
            for c in p.children(recursive=True):
                try:
                    cpu[c.pid] = c.cpu_percent(cfg.usage_update_interval)
                    mem[c.pid] = c.memory_info().rss / _MIB
                except psutil.NoSuchProcess:
                    cpu.pop(c.pid, None)
                    mem.pop(c.pid, None)
            # GPU
            if cfg.usage_gpu:
                if cfg.usage_gpu_force_torch or cls._pynvml_unavailable:
                    gpu = cls._gpu_torch(p.pid)
                else:
                    gpu = cls._gpu_nvml(p.pid, *cpu)
                    if gpu is None:
                        cls._pynvml_unavailable = True
                        gpu = cls._gpu_torch(p.pid)
            else:
                gpu = {}
            end_t = time.time_ns()
            cls._records_local.append(
                {"time": (start_t + end_t) // 2, "cpu": cpu, "memory": mem, "gpu": gpu}
            )
            remain_t = cfg.usage_update_interval - (end_t - start_t) / 1e9
            if remain_t > 0:
                time.sleep(remain_t)

    @classmethod
    def _gpu_nvml(cls, *pids: int) -> dict[int, float]:
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
            pids = set(pids)
            for handle in cls._pynvml_handles:
                for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    pid, mem = p.pid, p.usedGpuMemory
                    if (pid in pids) and (mem is not None):
                        gpu[pid] += mem / _MIB
            return gpu
        return {}

    @classmethod
    def _gpu_torch(cls, pid: int) -> dict[int, float]:
        import torch

        if cls._n_gpu is None:
            cls._n_gpu = torch.cuda.device_count()
        if cls._n_gpu > 0:
            if cls._torch_calibration is None:
                cls._torch_calibration = _CUDA.get(torch.version.cuda, 0)
            gpu = 0.0
            for i in range(cls._n_gpu):
                reserved = torch.cuda.memory_reserved(i)
                if reserved > 0:
                    gpu += (reserved / _MIB) + cls._torch_calibration
            return {pid: gpu}
        return {}

    @classmethod
    def serialize(cls):
        import json

        output = defaultdict(defaultdict[dict])
        for node in set(cls._checkpoints).intersection(cls._records):
            output[node[0]][node[1]] = {
                "checkpoints": cls._checkpoints[node],
                "records": cls._records[node],
            }
        return json.dumps(output).encode()


def setup_reporter():
    if cfg.usage_enable:
        Usage.start()


def setup_monitor():
    if cfg.usage_enable:
        Usage.start()
        Recorder.to_dump(cfg.file_usage, Usage.serialize)
