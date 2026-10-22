"""The GPUs of this host, as a GpuProvider.

Purpose and design. Asks nvidia-smi for the memory in use on every GPU of this host; the GPUs are
free when every one uses less than `threshold_mib` (a GPU with a process of anyone holds more).

Public components
    LocalGpus(threshold_mib=100)
        hosts() -> [this host's name], gpus() -> the number of GPUs, free() -> bool.

Example
    provider = LocalGpus()
    provider.free()
"""
import socket
import subprocess

from explicit_trainer.loop.abstract.gpu_provider import GpuProvider


def _memory_used_mib() -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [int(x) for x in out.split()]


class LocalGpus(GpuProvider):
    def __init__(self, threshold_mib: int = 100):
        self.threshold_mib = threshold_mib

    def hosts(self) -> list[str]:
        return [socket.gethostname()]

    def gpus(self) -> int:
        return len(_memory_used_mib())

    def free(self) -> bool:
        return all(used < self.threshold_mib for used in _memory_used_mib())
