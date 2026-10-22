"""The steps of a short time range of a profiled step with their real times and the graph's
dependencies between them, to see what every wait of the GPU waits for.

Purpose and design. `info` is framework.schedule.describe(graph) (graph.json); the trace is a
profiled step of the same graph (indexed annotations, steptrace.py; times in ms from the first
kernel of the compute stream, as the other tools). Rows: the host thread (when each step was
launched), the compute stream, and one row per stream of collectives and of host copies. Each step
is a bar from its first to its last kernel, labeled with its micro-batch (m0, m1, ...) and kind.
Arrows join a step to the steps it depends on that are drawn too: red when the step started right
after that dependency ended (it waited for it), grey otherwise; an orange tick marks a step that
started right after the host launched it (it waited for the host). The text form lists the same
steps in start order with what each waited for.

Public components
    TIGHT_MS
        A start this close after an end counts as waiting for it.
    steps(info, trace, start_ms, end_ms) -> dict
        The steps drawn: "shown" (step indices), "dev" ({step: [start ms, end ms, stream tid]}),
        "host" ({step: (start ms, end ms)} of its launch), "label" ({step: text}).
    text(info, trace, start_ms, end_ms) -> str
        One line per step: start, length, stream, launch, index, label, what it waited for.
    draw(info, trace, start_ms, end_ms, path)
        The chart as a PNG.

Example
    info, t = json.load(open("graph.json")), Trace("trace_rank0.json.gz")
    print(text(info, t, 1670, 1700))
    draw(info, t, 1670, 1700, "deps.png")
"""
from matplotlib import colormaps
from matplotlib.figure import Figure

from explicit_trainer.tools.steptrace import Trace

TIGHT_MS = 0.05


def _short(kind: str) -> str:
    """A step kind ("<file>.<class>.<function>") as "<class> <function>"."""
    parts = kind.split(".")
    if len(parts) < 2:
        return kind
    return f"{parts[-2]} {parts[-1].lstrip('_')}" if parts[-2][0].isupper() else parts[-1].lstrip("_")


def steps(info: dict, t: Trace, start_ms: float, end_ms: float) -> dict:
    def ms(us):
        return (us - t.t0) / 1e3

    host, dev = {}, {}
    for start, end, _, _, i in t.steps:
        if i >= 0:
            host[i] = (ms(start), ms(end))
    for k in t.kernels:
        j = t.step_of(k)
        i = t.steps[j][4] if j >= 0 else -1
        if i >= 0:
            d = dev.setdefault(i, [ms(k["ts"]), ms(k["ts"] + k["dur"]), k["tid"]])
            d[0], d[1] = min(d[0], ms(k["ts"])), max(d[1], ms(k["ts"] + k["dur"]))
    shown = sorted(i for i, (s, e, _) in dev.items() if e >= start_ms and s <= end_ms)
    label = {}
    for i in shown:
        mb = info["steps"][i]["micro_batch"]
        label[i] = (f"m{mb} " if mb is not None else "") + _short(info["steps"][i]["kind"])
    return dict(shown=shown, dev=dev, host=host, label=label)


def _waits(info: dict, found: dict, i: int) -> list[int]:
    s, dev = found["dev"][i][0], found["dev"]
    return [d for d in info["steps"][i]["deps"] if d in found["label"] and 0 <= s - dev[d][1] < TIGHT_MS]


def text(info: dict, t: Trace, start_ms: float, end_ms: float) -> str:
    found = steps(info, t, start_ms, end_ms)
    dev, host, label = found["dev"], found["host"], found["label"]
    lines = []
    for i in sorted(found["shown"], key=lambda i: dev[i][0]):
        s, e, tid = dev[i]
        waits = [f"#{d} {label[d]}" for d in _waits(info, found, i)]
        by_host = i in host and 0 <= s - host[i][1] < TIGHT_MS
        launched = f"{host[i][1]:9.2f}" if i in host else f"{'-':>9s}"
        lines.append(f"{s:9.2f} {e - s:6.2f} {'C' if tid == t.compute else tid:>6} launched {launched}  #{i:<6d} "
                     f"{label[i]:30s} waited for: {', '.join(waits) or ('host' if by_host else '-')}")
    return "\n".join(lines)


def draw(info: dict, t: Trace, start_ms: float, end_ms: float, path: str):
    found = steps(info, t, start_ms, end_ms)
    dev, host, label, shown = found["dev"], found["host"], found["label"], found["shown"]
    streams = sorted({dev[i][2] for i in shown}, key=lambda tid: (tid != t.compute, tid))
    rows = ["host"] + ["GPU compute" if tid == t.compute else f"GPU stream {tid}" for tid in streams]
    y = {tid: 1 + n for n, tid in enumerate(streams)}
    fig = Figure(figsize=(24, 2 + 1.6 * len(rows)))
    ax = fig.subplots()
    cmap = colormaps["tab10"]
    for i in shown:
        s, e, tid = dev[i]
        mb = info["steps"][i]["micro_batch"]
        color = cmap(mb % 10) if mb is not None else (0.6, 0.6, 0.6)
        ax.barh(y[tid], e - s, left=s, height=0.6, color=color, alpha=0.8, edgecolor="black", linewidth=0.3)
        ax.text((s + e) / 2, y[tid] + 0.38 * (1 if i % 2 else -1), label[i], ha="center", va="center", fontsize=6)
        if i in host:
            hs, he = host[i]
            ax.barh(0, max(he - hs, 0.02), left=hs, height=0.6, color=color, alpha=0.8, edgecolor="black",
                    linewidth=0.3)
            if 0 <= s - he < TIGHT_MS:
                ax.plot([s, s], [y[tid] - 0.45, y[tid] + 0.45], color="orange", linewidth=2)
        tight = set(_waits(info, found, i))
        for d in info["steps"][i]["deps"]:
            if d in label:
                ax.annotate("", xy=(s, y[tid]), xytext=(dev[d][1], y[dev[d][2]]),
                            arrowprops=dict(arrowstyle="->", color="red" if d in tight else "grey",
                                            lw=1.2 if d in tight else 0.5, alpha=0.9 if d in tight else 0.4))
    ax.set_yticks(range(len(rows)), rows)
    ax.invert_yaxis()
    ax.set_xlim(start_ms, end_ms)
    ax.set_xlabel("ms from the first kernel of the compute stream")
    ax.set_title("steps by micro-batch (color) and kind; red arrow: waited for that dependency; orange tick: waited "
                 "for the host launch")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
