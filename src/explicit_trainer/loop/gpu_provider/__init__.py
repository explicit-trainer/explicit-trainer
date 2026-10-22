"""Implementations of GpuProvider (explicit_trainer.loop.abstract.gpu_provider).

Public components
    LocalGpus       local.py: the GPUs of this host, through nvidia-smi
"""
from explicit_trainer.loop.gpu_provider.local import LocalGpus

__all__ = ["LocalGpus"]
