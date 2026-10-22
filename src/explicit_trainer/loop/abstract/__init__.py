"""The abstract classes a loop (explicit_trainer.loop.loop) is built from; the user passes implementations.

Public components
    GpuProvider     gpu_provider.py: the GPUs the agent may reserve, and whether they are free
    Checker         checker.py: whether a run achieves the goal
    Hook            hook.py: who else sees every message of the run
"""
from explicit_trainer.loop.abstract.checker import Checker
from explicit_trainer.loop.abstract.gpu_provider import GpuProvider
from explicit_trainer.loop.abstract.hook import Hook

__all__ = ["Checker", "GpuProvider", "Hook"]
