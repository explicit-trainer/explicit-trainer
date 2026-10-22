"""Whether a loop's goal is achieved.

Purpose and design. When the agent of a loop (loop.py) says it is done, it names the directory of
the run that shows it; a checker decides whether that run achieves the goal and explains why not.
What counts (a step time, a loss, an error level) and how it is verified (reading a result file,
rerunning on held-out data) belong to the goal, so the user passes an implementation to the loop.

Public components
    Checker
        check(directory) -> (achieved, output)
            achieved: bool; output: text for the agent and the log (why not, what was measured).

Example
    class StepTime(Checker):
        def check(self, directory):
            step_s = json.load(open(os.path.join(directory, "results.json")))["step_s"]
            return step_s <= 16.0, f"step_s {step_s:.2f}"
"""
from abc import ABC, abstractmethod


class Checker(ABC):
    @abstractmethod
    def check(self, directory: str) -> tuple[bool, str]:
        ...
