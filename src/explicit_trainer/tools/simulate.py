"""Designing schedules offline: estimate how a training step runs under a schedule, with step times
profiled under any schedule of the same graph.

Purpose and design. `info` is framework.schedule.describe(graph) of the plan's graph (written as
graph.json next to the schedules a plan's generator writes). The profile is the trace of one rank of
a step run with annotations (steptrace.py); it may come from another schedule of the same graph,
since step times are taken per step index: host time = the step's annotation minus the time it was
blocked inside CUDA calls (not host work), device time = from its first to its last kernel (a
collective includes its wait for the slowest rank, as measured). The simulation follows the trainer
exactly (framework/schedule.py): one host thread launches the head of any queue whose dependencies
are launched and whose to_host values have arrived, each launch costing the step's host time, and
when no head can go, waits for the first such value; a device step starts at max(its launch, its
stream free, the end of its dependencies on other device streams) and lasts its device time. Every
start records what bound it: the host (launch, or a wait for a to_host value), the step before it on
its stream, or a dependency. Not modelled: other ranks (their skew is in the measured collectives),
compute slowed by concurrent collectives, the CUDA launch queue limit, memory.

The report gives the simulated step, busy and idle time per stream, the idle time of each stream by
what the next step waited for (a dependency's kind and stream, the host's launch, the host waiting
for a to_host value), the N largest gaps of the compute stream, and the critical path (the chain of
bounds back from the last step) by kind and stream.

Public components
    durations(trace, n) -> (host, device)
        Per step index, seconds, from a steptrace.Trace.
    simulate(info, schedule, host, device) -> dict
        Returns "step_s", "t0", per step index "start", "end", "stream", "bound" (("host", step or
        None) | ("value", to_host step) | ("stream", step) | ("dep", step)), and "host_wait_s" (the
        host waiting for to_host values).
    idle(info, sim) -> {stream: {cause: s}}, gaps(info, sim, stream) -> [(s, at s, step, cause)]
    critical(info, sim) -> [step]
        The critical path, last step first.
    report(info, sim, largest=10) -> str
    around(info, sim, ms, span) -> str
        The simulated steps starting within ms +- span, with what bound each.
    draw(info, sim, path, lo_ms, hi_ms)
        The simulated step as a PNG: rows host, then each stream; bars colored by micro-batch.

Example
    info = json.load(open("graph.json"))
    host, device = durations(Trace("trace_rank0.json.gz"), len(info["steps"]))
    sim = simulate(info, json.load(open("schedule.json")), host, device)
    print(report(info, sim))
"""
import collections

from matplotlib import colormaps
from matplotlib.figure import Figure

from explicit_trainer.tools.steptrace import Trace
_COPY_S = 20e-6  # device time of a to_host copy (not in the kernel list of a trace)


def durations(trace: Trace, n: int) -> tuple[list[float], list[float]]:
    host, first, last = [0.0] * n, [None] * n, [0.0] * n
    for i, (start, end, _, _, index) in enumerate(trace.steps):
        if 0 <= index < n:
            host[index] += (end - start - trace.blocked(i)) / 1e6
    for k in trace.kernels:
        i = trace.step_of(k)
        index = trace.steps[i][4] if i >= 0 else -1
        if 0 <= index < n:
            first[index] = k["ts"] if first[index] is None else min(first[index], k["ts"])
            last[index] = max(last[index], k["ts"] + k["dur"])
    return host, [(last[i] - first[i]) / 1e6 if first[i] is not None else 0.0 for i in range(n)]


def simulate(info: dict, schedule: dict, host: list[float], device: list[float]) -> dict:
    steps, n = info["steps"], len(info["steps"])
    start, end, stream, bound = [0.0] * n, [0.0] * n, [None] * n, [None] * n
    launched = [False] * n
    for q, members in schedule["queues"].items():
        for i in members:
            stream[i] = q
    pending = {q: collections.deque(members) for q, members in schedule["queues"].items()}
    arrivals = [[d for d in s["deps"] if steps[d]["type"] == "to_host"] for s in steps]
    clock, wait, free, by_host = 0.0, 0.0, {}, ("host", None)
    while any(pending.values()):
        moved = False
        for q, queue in pending.items():
            while queue and all(launched[d] for d in steps[queue[0]]["deps"]) and \
                    all(end[d] <= clock for d in arrivals[queue[0]]):
                i = queue.popleft()
                launched[i], moved = True, True
                if steps[i]["type"] != "input":
                    clock += host[i]
                if q == "host":
                    start[i], end[i], bound[i] = clock - host[i], clock, by_host
                else:
                    options = [(clock, by_host)] + ([(end[free[q]], ("stream", free[q]))] if q in free else []) + \
                        [(end[d], ("dep", d)) for d in steps[i]["deps"] if stream[d] not in ("host", q)]
                    start[i], bound[i] = max(options, key=lambda o: o[0])
                    end[i] = start[i] + (_COPY_S if q == "to_host" else device[i])
                    free[q] = i
                by_host = ("host", i)
        if not moved:
            needed = [d for queue in pending.values() if queue and all(launched[x] for x in steps[queue[0]]["deps"])
                      for d in arrivals[queue[0]] if end[d] > clock]
            assert needed, "no queue can go and none waits for a host value"
            d = min(needed, key=lambda d: end[d])
            wait += end[d] - clock
            clock, by_host = end[d], ("value", d)
    t0 = min(start[i] for i in range(n) if stream[i] != "host" and end[i] > start[i])
    return dict(step_s=max(end) - t0, t0=t0, start=start, end=end, stream=stream, bound=bound, host_wait_s=wait)


def _cause(info: dict, sim: dict, i: int) -> str:
    how, d = sim["bound"][i]
    if how == "dep":
        return f"{info['steps'][d]['kind']} on {sim['stream'][d]}"
    return {"host": "host launch", "value": "host waits for a to_host value", "stream": "-"}[how]


def _on(sim: dict, q: str) -> list[int]:
    return sorted((i for i, s in enumerate(sim["stream"]) if s == q), key=lambda i: sim["start"][i])


def gaps(info: dict, sim: dict, q: str) -> list:
    out, last = [], None
    for i in _on(sim, q):
        if last is not None and sim["start"][i] > last:
            out.append((sim["start"][i] - last, last - sim["t0"], i, _cause(info, sim, i)))
        last = sim["end"][i] if last is None else max(last, sim["end"][i])
    return out


def idle(info: dict, sim: dict) -> dict:
    out = {}
    for q in sorted({s for s in sim["stream"] if s != "host"}):
        causes = collections.Counter()
        for g, _, _, cause in gaps(info, sim, q):
            causes[cause] += g
        out[q] = dict(causes)
    return out


def critical(info: dict, sim: dict) -> list[int]:
    i = max(range(len(sim["end"])), key=lambda k: sim["end"][k])
    path = []
    while i is not None:
        path.append(i)
        i = sim["bound"][i][1]
    return path


def report(info: dict, sim: dict, largest: int = 10) -> str:
    span = sim["step_s"]
    lines = [f"simulated step {span * 1e3:.1f} ms; host waited {sim['host_wait_s'] * 1e3:.1f} ms for to_host values"]
    for q, causes in idle(info, sim).items():
        busy = sum(sim["end"][i] - sim["start"][i] for i in _on(sim, q))
        total = sum(causes.values())
        lines.append(f"  {q:10s} busy {busy * 1e3:8.1f} ms, idle between steps {total * 1e3:8.1f} ms")
        for cause, s in sorted(causes.items(), key=lambda kv: -kv[1])[:6]:
            lines.append(f"      {s * 1e3:8.1f} ms  waiting for {cause[-70:]}")
    lines.append("  largest gaps of the compute stream (ms, at ms, step after, what it waited for)")
    for g, at, i, cause in sorted(gaps(info, sim, "compute"), reverse=True)[:largest]:
        s = info["steps"][i]
        lines.append(f"    {g * 1e3:6.2f} at {at * 1e3:8.1f}  #{i} m{s['micro_batch']} {s['kind'][-40:]}  "
                     f"<- {cause[-50:]}")
    by = collections.Counter()
    for i in critical(info, sim):
        by[sim["stream"][i], info["steps"][i]["kind"]] += sim["end"][i] - sim["start"][i]
    lines.append("  critical path by stream and kind (ms)")
    for (q, k), s in by.most_common(10):
        lines.append(f"    {s * 1e3:8.1f}  {q:10s} {k[-60:]}")
    return "\n".join(lines)


def around(info: dict, sim: dict, ms: float, span: float) -> str:
    lines = [f"=== around {ms} ms (start, length, stream, step, bound)"]
    for i in sorted(range(len(sim["start"])), key=lambda k: sim["start"][k]):
        x = (sim["start"][i] - sim["t0"]) * 1e3
        if ms - span <= x <= ms + span:
            s = info["steps"][i]
            lines.append(f"  {x:9.2f} {(sim['end'][i] - sim['start'][i]) * 1e3:6.2f} {sim['stream'][i]:9s} #{i:<6d} "
                         f"m{s['micro_batch']} {s['kind'][-40:]:40s} <- {_cause(info, sim, i)[-50:]}")
    return "\n".join(lines)


def draw(info: dict, sim: dict, path: str, lo: float, hi: float):
    rows = ["host"] + sorted({q for q in sim["stream"] if q != "host"}, key=lambda q: (q != "compute", q))
    fig = Figure(figsize=(22, 1.0 + 0.6 * len(rows)))
    ax = fig.subplots()
    cmap = colormaps["tab10"]
    for i, q in enumerate(sim["stream"]):
        a, b = (sim["start"][i] - sim["t0"]) * 1e3, (sim["end"][i] - sim["t0"]) * 1e3
        if b < lo or a > hi or b <= a:
            continue
        mb = info["steps"][i]["micro_batch"]
        ax.barh(rows.index(q), b - a, left=a, height=0.7, color=cmap(mb % 10) if mb is not None else (0.6, 0.6, 0.6),
                edgecolor="black", linewidth=0.2)
    ax.set_yticks(range(len(rows)), rows)
    ax.invert_yaxis()
    ax.set_xlim(lo, hi)
    ax.set_xlabel("simulated ms (color: micro-batch, grey: none)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
