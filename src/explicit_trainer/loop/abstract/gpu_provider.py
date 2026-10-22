"""Where a loop's agent may run: the GPUs it reserves, and whether they are free.

Purpose and design. A loop (loop.py) reserves GPUs for its agent only when they are free: when no
process of anyone runs on them. Which machines those are and how to look at them depends on the
environment (this host, a cluster behind a gateway, a scheduler), so it is an implementation of
this class that the user passes to the loop. A reservation is all of the provider's GPUs.

Public components
    GpuProvider
        hosts() -> [str]
            The machines the agent may use, by the names the agent is told.
        gpus() -> int
            How many GPUs they have in all (for GPU-hours).
        free() -> bool
            Whether every GPU of every host is free now.

Example
    class OneHost(GpuProvider):
        def hosts(self):
            return ["localhost"]

        def gpus(self):
            return 8

        def free(self):
            return all(used < 100 for used in memory_used_mib())
"""
from abc import ABC, abstractmethod


class GpuProvider(ABC):
    @abstractmethod
    def hosts(self) -> list[str]:
        ...

    @abstractmethod
    def gpus(self) -> int:
        ...

    @abstractmethod
    def free(self) -> bool:
        ...
