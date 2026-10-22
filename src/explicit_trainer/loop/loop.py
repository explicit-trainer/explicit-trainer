"""The loop: the scripted side of an agent's run toward a goal. It answers every message of the agent
with fixed text, keeps the run's state and log, reserves GPUs, and decides when the goal is achieved.

Purpose and design. The loop is passive: it never calls the agent. Whoever runs the agent relays
each of its messages to reply() and the answer back (directly, or through server.py), so the agent
cannot bypass the protocol, and the log is the record of the run. What the environment provides
comes in as implementations of the abstract classes: the GPUs (GpuProvider), the goal's check
(Checker), and who else sees every message (Hook: called after each is logged). The agent's protocol, a line of its message starting with a keyword in capitals (the
first match wins):
    RESERVE <n> GPUs <minutes>  -> READY <hosts> if every GPU of the provider is free, else WAIT
    RELEASE                     -> RELEASED, and the instructions again
    PAUSE <reason>              -> PAUSED; the run waits until resume() (the environment failed)
    DONE <run directory>        -> ACHIEVED (the run ends) if the checker says so, else NOT ACHIEVED
                                   with the checker's output
    anything else               -> CONTINUE_TEXT, with the instructions every RESEND_EVERY such replies
The instructions (the goal's text and PROTOCOL_TEXT) are sent at start, at every resume and after
every RELEASE. Every message in and out is appended to <run dir>/log.jsonl with its time;
<run dir>/state.json holds the phase, the reservation, the GPU-hours, the pauses, the checks and the
counts, so a loop made again on the same directory continues the run. The active time of a run
excludes pauses and waits for READY.

Public components
    CONTINUE_TEXT, RESEND_EVERY, PROTOCOL_TEXT
    Loop(run_dir, instructions, provider, checker, hooks=())
        start() -> str          the first message to the agent (the run must not have started)
        reply(text) -> str      the answer to one message of the agent
        resume() -> str         after a pause: the message that resumes the agent
        summary() -> dict       the state, with "wall_s" and "active_s"

Example
    loop = Loop("runs/r1", open("goal.md").read(), LocalGpus(), CommandChecker('python check.py "$1"'))
    print(loop.start())
    print(loop.reply("RESERVE 8 GPUs 30"))      # READY <host> or WAIT
"""
import json
import os
import re
import time

from explicit_trainer.loop.abstract.checker import Checker
from explicit_trainer.loop.abstract.gpu_provider import GpuProvider
from explicit_trainer.loop.abstract.hook import Hook

CONTINUE_TEXT = "Continue."
RESEND_EVERY = 10
PROTOCOL_TEXT = """Protocol (in capitals at the start of a line, at most one per message; everything else is free):
RESERVE <n> GPUs <minutes>   before using GPUs; wait for READY (WAIT: ask again later)
RELEASE                      when you no longer use them
PAUSE <reason>               when the environment fails (a machine down, GPUs taken by others);
                             then wait for RESUME
DONE <run directory>         when you believe the goal is achieved; the directory of the run that shows it"""


class Loop:
    def __init__(self, run_dir: str, instructions: str, provider: GpuProvider, checker: Checker,
                 hooks: tuple[Hook, ...] = ()):
        self.run_dir, self.provider, self.checker, self.hooks = run_dir, provider, checker, hooks
        self.instructions = instructions.strip() + "\n\n" + PROTOCOL_TEXT
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "state.json")
        self.state = json.load(open(path)) if os.path.exists(path) else None

    def _save(self):
        json.dump(self.state, open(os.path.join(self.run_dir, "state.json"), "w"), indent=1)

    def _log(self, direction: str, text: str):
        with open(os.path.join(self.run_dir, "log.jsonl"), "a") as f:
            f.write(json.dumps({"t": time.time(), "dir": direction, "text": text}) + "\n")
        for hook in self.hooks:
            hook.message(direction, text)

    def _send(self, text: str) -> str:
        self._save()
        self._log("loop", text)
        return text

    def start(self) -> str:
        assert self.state is None, "the run has already started"
        self.state = {"phase": "running", "start": time.time(), "end": None, "messages": 0, "continues": 0,
                      "reservation": None, "gpu_hours": 0.0, "waiting_s": 0.0, "wait_since": None, "pauses": [],
                      "checks": []}
        return self._send(self.instructions)

    def reply(self, text: str) -> str:
        s = self.state
        assert s is not None and s["phase"] != "achieved", "the run is not running"
        self._log("agent", text)
        s["messages"] += 1
        if s["phase"] == "paused":
            return self._send("PAUSED. Wait for RESUME.")
        if m := re.search(r"^\s*RESERVE\s+(\d+)\s+GPUs?\s+(\d+)", text, re.M):
            return self._send(self._reserve(int(m.group(1)), int(m.group(2))))
        if re.search(r"^\s*RELEASE\b", text, re.M):
            self._release()
            return self._send("RELEASED\n\n" + self.instructions)
        if m := re.search(r"^\s*PAUSE\b\s*(.*)$", text, re.M):
            s["phase"] = "paused"
            s["pauses"].append({"since": time.time(), "until": None, "reason": m.group(1).strip()})
            return self._send("PAUSED. Wait for RESUME.")
        if m := re.search(r"^\s*DONE\s+(\S+)", text, re.M):
            achieved, output = self.checker.check(m.group(1))
            s["checks"].append({"t": time.time(), "dir": m.group(1), "achieved": achieved, "out": output[-2000:]})
            if achieved:
                s["phase"], s["end"] = "achieved", time.time()
                return self._send("ACHIEVED. The run is over; stop working.")
            return self._send("NOT ACHIEVED\n" + output[-1500:] + "\n\n" + CONTINUE_TEXT)
        s["continues"] += 1
        out = CONTINUE_TEXT
        if s["continues"] % RESEND_EVERY == 0:
            out += "\n\n" + self.instructions
        return self._send(out)

    def _reserve(self, gpus: int, minutes: int) -> str:
        s, hosts = self.state, self.provider.hosts()
        if s["reservation"] is not None:
            return "READY " + " ".join(hosts) + " (already reserved)"
        if s["wait_since"] is None:
            s["wait_since"] = time.time()
        if not self.provider.free():
            return "WAIT"
        s["waiting_s"] += time.time() - s["wait_since"]
        s["wait_since"] = None
        s["reservation"] = {"hosts": hosts, "since": time.time(), "gpus": gpus, "minutes": minutes}
        return "READY " + " ".join(hosts)

    def _release(self):
        r = self.state["reservation"]
        if r is not None:
            self.state["gpu_hours"] += (time.time() - r["since"]) / 3600 * self.provider.gpus()
            self.state["reservation"] = None

    def resume(self) -> str:
        assert self.state is not None and self.state["phase"] == "paused", "the run is not paused"
        self.state["phase"] = "running"
        self.state["pauses"][-1]["until"] = time.time()
        return self._send("RESUME\n\n" + self.instructions)

    def summary(self) -> dict:
        s = self.state
        assert s is not None, "the run has not started"
        end = s["end"] or time.time()
        paused = sum((p["until"] or end) - p["since"] for p in s["pauses"])
        return s | {"wall_s": end - s["start"], "active_s": end - s["start"] - paused - s["waiting_s"]}
