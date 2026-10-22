"""Who else sees a loop's messages.

Purpose and design. A loop (loop.py) writes every message it receives from the agent and every answer it sends to
its log; a hook is called right after each, so that people or other programs can follow a run as it happens (e.g.
a chat that shows every message). A hook observes only: it cannot change the message or the answer.

Public components
    Hook
        message(direction, text)
            direction: "agent" for a message the loop received, "loop" for an answer it sent; text: the message.

Example
    class Printer(Hook):
        def message(self, direction, text):
            print(f"[{direction}] {text}")
"""
from abc import ABC, abstractmethod


class Hook(ABC):
    @abstractmethod
    def message(self, direction: str, text: str):
        ...
