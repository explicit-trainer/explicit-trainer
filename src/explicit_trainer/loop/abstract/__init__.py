"""The abstract classes a loop (explicit_trainer.loop.loop) is built from; the user passes implementations.

Public components
    GpuProvider     gpu_provider.py: the GPUs the agent may reserve, and whether they are free
    Checker         checker.py: whether a run achieves the goal
"""
from explicit_trainer.loop.abstract.checker import Checker
from explicit_trainer.loop.abstract.gpu_provider import GpuProvider

__all__ = ["Checker", "GpuProvider"]
