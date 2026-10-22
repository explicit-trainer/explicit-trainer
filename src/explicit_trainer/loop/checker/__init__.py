"""Implementations of Checker (explicit_trainer.loop.abstract.checker).

Public components
    CommandChecker  command.py: a shell command on the run's directory; exit status 0 means achieved
"""
from explicit_trainer.loop.checker.command import CommandChecker

__all__ = ["CommandChecker"]
