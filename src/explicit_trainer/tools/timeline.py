"""What happens on every stream of a training step, step by step: as text around given times, or
as a Gantt chart.

Purpose and design. The text form lists, for each given time (ms from the first kernel of the
compute stream), every kernel within +-span ms on every stream in start order: start, duration,
stream (C: the compute stream, s<tid>: another stream, e.g. a collective's), when the host launched
it and inside which step, and the kernel's name. Collective kernels show their full duration,
including waiting for the other ranks. It answers what a gap waits for (use the gaps that gaps.py
lists). The chart draws the same information over a time range: one row per stream, the compute
stream first, then one row per host thread that runs steps; kernels and host steps are colored by
the kind of step that launched them (the 18 largest kinds, the rest in grey) and collective kernels
by collective; idle time is blank.

Public components
    around(trace, ms, span) -> [line]
        The text lines for one time.
    gantt(trace, path, start_ms, end_ms)
        Draws the chart to a PNG file.

Example
    t = Trace("trace_rank0.json.gz")
    print("\n".join(around(t, 2434.2, 3.0)))
    gantt(t, "gantt.png", 1800.0, 2200.0)
"""
import collections
import re

from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from explicit_trainer.tools.steptrace import Trace


def around(trace: Trace, ms: float, span: float) -> list:
    lines = [f"=== around {ms} ms"]
    for k in trace.kernels:
        x = (k["ts"] - trace.t0) / 1e3
        if ms - span <= x <= ms + span:
            launch = trace.launch(k)
            i = trace.step_of(k)
            stream = "C" if k["tid"] == trace.compute else f"s{k['tid']}"
            launched = f"{(launch[0] - trace.t0) / 1e3:9.2f}" if launch else f"{'-':>9s}"
            step = trace.steps[i][3] if i >= 0 else "-"
            lines.append(f"{x:9.2f} {k['dur'] / 1e3:6.2f} {stream:>4} launched {launched} {step[-40:]:40s} "
                         f"{k['name'][:50]}")
    return lines


def _label(trace: Trace, k) -> str:
    if k["tid"] != trace.compute and "nccl" in k["name"].lower():
        m = re.search(r"(AllGather|ReduceScatter|AllReduce|SendRecv|AllToAll|Broadcast)", k["name"])
        return f"nccl {m[1] if m else 'other'}"
    i = trace.step_of(k)
    return _short(trace.steps[i][3]) if i >= 0 else "(outside steps)"


def _short(kind: str) -> str:
    """The class and function of a step kind (the type of other steps)."""
    return ".".join(kind.split(".")[-2:])


def gantt(trace: Trace, path: str, start_ms: float, end_ms: float):
    bars = collections.defaultdict(list)
    for k in trace.kernels:
        row = "GPU compute" if k["tid"] == trace.compute else f"GPU stream {k['tid']}"
        bars[row].append((k["ts"], k["dur"], _label(trace, k)))
    threads = sorted({s[2] for s in trace.steps})
    for start, end, tid, name, _ in trace.steps:
        bars[f"host thread {threads.index(tid)}"].append((start, end - start, _short(name)))
    rows = ["GPU compute"] + sorted(r for r in bars if r.startswith("GPU stream")) + \
        sorted(r for r in bars if r.startswith("host"))
    weight = collections.Counter()
    for r in rows:
        for _, d, label in bars[r]:
            weight[label] += d
    top = [label for label, _ in weight.most_common(18)]
    color = {label: colormaps["tab20"](i) for i, label in enumerate(top)}
    grey = (0.6, 0.6, 0.6, 1)
    a, b = trace.t0 + start_ms * 1e3, trace.t0 + end_ms * 1e3
    fig = Figure(figsize=(22, 1.0 + 0.55 * len(rows)))
    ax = fig.subplots()
    for y, r in enumerate(rows):
        seg = collections.defaultdict(list)
        for s, d, label in bars[r]:
            if s + d >= a and s <= b:
                seg[label if label in color else "other"].append(((s - trace.t0) / 1e3, max(d, 0.5) / 1e3))
        for label, xs in seg.items():
            ax.broken_barh(xs, (y - 0.4, 0.8), facecolors=color.get(label, grey), linewidth=0)
    ax.set_yticks(range(len(rows)), rows)
    ax.invert_yaxis()
    ax.set_xlim(start_ms, end_ms)
    ax.set_xlabel("ms from the first kernel of the compute stream")
    ax.legend([Rectangle((0, 0), 1, 1, color=color[t]) for t in top] + [Rectangle((0, 0), 1, 1, color=grey)],
              top + ["other"], loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
