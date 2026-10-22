"""A shell command as a Hook.

Purpose and design. Runs `bash -c <command> hook <direction>` for every message: the command sees the direction
("agent" or "loop") as $1 and the message on its standard input, and has `timeout` seconds. A hook observes only, so
a command that fails or times out does not stop the loop: the failure is printed on the standard error of the
process that serves the loop, and the loop goes on.

Public components
    CommandHook(command, timeout=60)
        message(direction, text)

Example
    hook = CommandHook('microteams api say --thread-id 857 --text "[$1] $(cat)"')
"""
import subprocess
import sys

from explicit_trainer.loop.abstract.hook import Hook


class CommandHook(Hook):
    def __init__(self, command: str, timeout: float = 60):
        self.command, self.timeout = command, timeout

    def message(self, direction: str, text: str):
        try:
            r = subprocess.run(["bash", "-c", self.command, "hook", direction], input=text, capture_output=True,
                               text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            print(f"hook timed out after {self.timeout} s ({direction})", file=sys.stderr, flush=True)
            return
        if r.returncode != 0:
            print(f"hook failed ({direction}, status {r.returncode}): {r.stderr[-500:]}", file=sys.stderr, flush=True)
