"""A shell command as a Checker.

Purpose and design. Runs `bash -c <command> check <directory>`: the command sees the run's directory
as $1. Exit status 0 means achieved; its standard output and error are the explanation.

Public components
    CommandChecker(command)
        check(directory) -> (achieved, output)

Example
    checker = CommandChecker('python check_goal.py "$1"')
    achieved, output = checker.check("/runs/attempt-3")
"""
import subprocess

from explicit_trainer.loop.abstract.checker import Checker


class CommandChecker(Checker):
    def __init__(self, command: str):
        self.command = command

    def check(self, directory: str) -> tuple[bool, str]:
        r = subprocess.run(["bash", "-c", self.command, "check", directory], capture_output=True, text=True)
        return r.returncode == 0, r.stdout + r.stderr
