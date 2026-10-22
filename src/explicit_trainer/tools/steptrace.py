"""Reading a PyTorch profiler trace of one training step written with step annotations.

Purpose and design. The trainer, with `annotate` on, wraps every step in a profiler range named
"step <kind> #<index>": kind is "<file>.<qualified function name>" of a compute step, else the step's
type (e.g. AllToAll); index is its position in the graph's steps. This module turns the trace of
one rank into what the other tools (steps, gaps, timeline, simulate, deps, order) need: the
kernels in start order, which CUDA stream is the compute stream (the one with the most kernels),
the step annotations, and for every kernel the call of a step that launched it (the annotation open
on the launching thread when the CUDA runtime or driver launch call was made). Times are
microseconds as in the trace; the tools convert.

Public components
    Trace(path)
        Loads a trace (.json or .json.gz). Attributes:
        kernels: [event] all GPU kernels in start order; an event has ts, dur, tid, name, args.
        compute: the tid of the compute stream.
        steps: [(start, end, tid, kind, index)] the step annotations in start order.
        t0, t1: start of the first and end of the last kernel of the compute stream.
    Trace.launch(kernel) -> (ts, tid) or None
        When and on which host thread the kernel was launched.
    Trace.step_of(kernel) -> int
        Index into `steps` of the call that launched the kernel, or -1 (launched outside any step).
    Trace.on_compute() -> [event]
        The kernels of the compute stream in start order.
    Trace.blocked(i) -> float
        Microseconds the host thread of step annotation i spent inside CUDA runtime or driver calls
        longer than 50 us: waiting (e.g. a launch call blocked on a full launch queue), not host work.

Example
    t = Trace("trace_rank0.json.gz")
    for k in t.on_compute():
        i = t.step_of(k)
        print(k["name"], t.steps[i][3] if i >= 0 else "(outside steps)")
"""
import bisect
import collections
import gzip
import json

_LAUNCH = ("cuda_runtime", "cuda_driver")


def _kind_index(name: str) -> tuple[str, int]:
    kind, _, index = name.partition(" #")
    return kind, int(index)


class Trace:
    def __init__(self, path: str):
        opener = gzip.open if path.endswith(".gz") else open
        events = [e for e in json.load(opener(path))["traceEvents"] if e.get("ph") == "X"]
        self.kernels = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
        self.compute = collections.Counter(e["tid"] for e in self.kernels).most_common(1)[0][0]
        self.steps = sorted((e["ts"], e["ts"] + e["dur"], e["tid"], *_kind_index(e["name"][5:])) for e in events
                            if e.get("cat") == "user_annotation" and e.get("name", "").startswith("step "))
        self._starts = [s[0] for s in self.steps]
        self._launches = {e["args"]["correlation"]: (e["ts"], e["tid"]) for e in events
                          if e.get("cat") in _LAUNCH and "correlation" in e.get("args", {})}
        self._calls = collections.defaultdict(list)  # host thread -> [(start, duration)] of long CUDA API calls
        for e in events:
            if e.get("cat") in _LAUNCH and e["dur"] > 50:
                self._calls[e["tid"]].append((e["ts"], e["dur"]))
        for calls in self._calls.values():
            calls.sort()
        compute = self.on_compute()
        self.t0, self.t1 = compute[0]["ts"], max(k["ts"] + k["dur"] for k in compute)

    def launch(self, kernel):
        return self._launches.get(kernel.get("args", {}).get("correlation"))

    def blocked(self, i: int) -> float:
        start, end, tid = self.steps[i][:3]
        calls = self._calls.get(tid, [])
        j = bisect.bisect_left(calls, (start,))
        total = 0.0
        while j < len(calls) and calls[j][0] < end:
            total += calls[j][1]
            j += 1
        return total

    def step_of(self, kernel) -> int:
        launch = self.launch(kernel)
        if launch is None:
            return -1
        ts, tid = launch
        i = bisect.bisect_right(self._starts, ts) - 1
        while i >= 0 and self.steps[i][0] >= ts - 30e6:  # annotations nest at most this deep in time
            start, end, thread = self.steps[i][:3]
            if start <= ts <= end and thread == tid:
                return i
            i -= 1
        return -1

    def on_compute(self) -> list:
        return [k for k in self.kernels if k["tid"] == self.compute]
