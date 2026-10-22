"""The loop's protocol, state and log, and its socket server.

Public components
    FakeGpus
        A GpuProvider of two hosts and 16 GPUs whose `is_free` the test sets.
    Recorder
        A Hook that keeps every (direction, text) it is given.
    Exists
        A Checker: achieved when the directory exists.
    LoopTest

Example
    python -m unittest tests.test_loop
"""
import json
import os
import tempfile
import threading
import time
import unittest

from explicit_trainer.loop.abstract.checker import Checker
from explicit_trainer.loop.abstract.gpu_provider import GpuProvider
from explicit_trainer.loop.abstract.hook import Hook
from explicit_trainer.loop.checker.command import CommandChecker
from explicit_trainer.loop.hook.command import CommandHook
from explicit_trainer.loop.loop import CONTINUE_TEXT, RESEND_EVERY, Loop
from explicit_trainer.loop.server import request, serve


class FakeGpus(GpuProvider):
    def __init__(self):
        self.is_free = False

    def hosts(self):
        return ["a", "b"]

    def gpus(self):
        return 16

    def free(self):
        return self.is_free


class Exists(Checker):
    def check(self, directory):
        return os.path.isdir(directory), f"checked {directory}"


class Recorder(Hook):
    def __init__(self):
        self.seen = []

    def message(self, direction, text):
        self.seen.append((direction, text))


class LoopTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.gpus = FakeGpus()
        self.loop = Loop(os.path.join(self.dir, "run"), "Make it fast.", self.gpus, Exists())

    def test_protocol(self):
        first = self.loop.start()
        self.assertTrue(first.startswith("Make it fast."))
        self.assertIn("RESERVE", first)
        self.assertRaises(AssertionError, self.loop.start)
        self.assertEqual(self.loop.reply("thinking\nRESERVE 16 GPUs 30"), "WAIT")
        self.gpus.is_free = True
        self.assertEqual(self.loop.reply("RESERVE 16 GPUs 30"), "READY a b")
        self.assertEqual(self.loop.reply("RESERVE 16 GPUs 30"), "READY a b (already reserved)")
        self.assertTrue(self.loop.reply("RELEASE").startswith("RELEASED\n\nMake it fast."))
        self.assertEqual(self.loop.reply("PAUSE machine down"), "PAUSED. Wait for RESUME.")
        self.assertEqual(self.loop.reply("hello"), "PAUSED. Wait for RESUME.")
        self.assertTrue(self.loop.resume().startswith("RESUME"))
        self.assertRaises(AssertionError, self.loop.resume)
        for k in range(1, RESEND_EVERY + 1):
            out = self.loop.reply("working")
            self.assertEqual(out == CONTINUE_TEXT, k < RESEND_EVERY)
        self.assertTrue(self.loop.reply("DONE /no/such/dir").startswith("NOT ACHIEVED\nchecked /no/such/dir"))
        self.assertEqual(self.loop.reply(f"DONE {self.dir}"), "ACHIEVED. The run is over; stop working.")
        self.assertRaises(AssertionError, self.loop.reply, "more")
        s = self.loop.summary()
        self.assertEqual(s["phase"], "achieved")
        self.assertEqual(len(s["checks"]), 2)
        self.assertGreater(s["gpu_hours"], 0)
        self.assertLessEqual(s["active_s"], s["wall_s"])
        log = [json.loads(line) for line in open(os.path.join(self.dir, "run", "log.jsonl"))]
        self.assertEqual(log[0]["dir"], "loop")
        self.assertEqual(sum(e["dir"] == "agent" for e in log), s["messages"])

    def test_state_persists(self):
        self.loop.start()
        self.loop.reply("PAUSE down")
        again = Loop(os.path.join(self.dir, "run"), "Make it fast.", self.gpus, Exists())
        self.assertEqual(again.summary()["phase"], "paused")
        self.assertTrue(again.resume().startswith("RESUME"))

    def test_hooks(self):
        rec = Recorder()
        out = os.path.join(self.dir, "hook.txt")
        failing = CommandHook("exit 3")
        loop = Loop(os.path.join(self.dir, "hooked"), "Goal.", self.gpus, Exists(),
                    hooks=(rec, CommandHook(f'printf "[%s] %s\\n" "$1" "$(cat)" >> {out}'), failing))
        first = loop.start()
        answer = loop.reply("RESERVE 16 GPUs 5")
        self.assertEqual(rec.seen, [("loop", first), ("agent", "RESERVE 16 GPUs 5"), ("loop", answer)])
        self.assertEqual(open(out).read().splitlines()[-2:], ["[agent] RESERVE 16 GPUs 5", "[loop] WAIT"])

    def test_command_checker(self):
        self.assertEqual(CommandChecker('test -d "$1" && echo yes').check(self.dir), (True, "yes\n"))
        self.assertFalse(CommandChecker('test -d "$1"').check("/no/such/dir")[0])

    def test_server(self):
        path = os.path.join(self.dir, "loop.sock")
        threading.Thread(target=serve, args=(self.loop, path), daemon=True).start()
        while not os.path.exists(path):
            time.sleep(0.01)
        self.assertIn("Make it fast.", request(path, "start"))
        self.assertEqual(request(path, "msg", "RESERVE 16 GPUs 5"), "WAIT")
        self.assertEqual(request(path, "state")["phase"], "running")
        with self.assertRaises(RuntimeError):
            request(path, "resume")


if __name__ == "__main__":
    unittest.main()
