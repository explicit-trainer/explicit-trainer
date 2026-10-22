"""Implementations of Hook (explicit_trainer.loop.abstract.hook).

Public components
    CommandHook     command.py: a shell command run for every message, with the message on its standard input
"""
from explicit_trainer.loop.hook.command import CommandHook

__all__ = ["CommandHook"]
