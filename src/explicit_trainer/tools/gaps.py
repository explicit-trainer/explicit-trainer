"""Where and why the compute stream of a training step is idle.

Purpose and design. Every gap between consecutive kernels of the compute stream is classified:
in-step when the kernels on both sides belong to the same call of a compute step (time lost between
the kernels of one step: their number and the device's gap between dependent kernels), out-step
otherwise (the stream waits between steps: for a collective, for the host, or for nothing to be
launched). The report gives the totals of both, the in-step idle per kind of step, the out-step
gaps by size, the N largest out-step gaps with the steps before and after them and the collective
(kernel on another stream) that ended last before the next kernel started, the idle time in each
twentieth of the step, and how far the host runs ahead of the GPU there (kernel start minus launch:
a lead near zero means the GPU waits for the host). The report is plain text.

Public components
    gaps(trace) -> [(start us, length us, step before, step after, ending collective or "-")]
        Every gap of the compute stream, in order; steps as indices into trace.steps (-1: none).
    report(trace, largest=15) -> str
        The report.

Example
    print(report(Trace("trace_rank0.json.gz"), largest=20))
"""
import bisect
import collections

from explicit_trainer.tools.steptrace import Trace


def gaps(trace: Trace) -> list:
    other = sorted((k for k in trace.kernels if k["tid"] != trace.compute), key=lambda k: k["ts"] + k["dur"])
    ends = [k["ts"] + k["dur"] for k in other]
    out, end, prev = [], None, -1
    for k in trace.on_compute():
        step = trace.step_of(k)
        if end is not None and k["ts"] > end:
            j = bisect.bisect_right(ends, k["ts"]) - 1
            ending = other[j]["name"] if j >= 0 and ends[j] > end else "-"
            out.append((end, k["ts"] - end, prev, step, ending))
        if end is None or k["ts"] + k["dur"] > end:
            end, prev = k["ts"] + k["dur"], step
    return out


def report(t: Trace, largest: int = 15) -> str:
    span = (t.t1 - t.t0) / 1e3
    found = gaps(t)

    def name(i):
        return t.steps[i][3] if i >= 0 else "(none)"

    inside = collections.Counter()
    outside = []
    for start, length, before, after, ending in found:
        if before >= 0 and before == after:
            inside[name(before)] += length / 1e3
        else:
            outside.append((length / 1e3, (start - t.t0) / 1e3, name(before), name(after), ending))
    total_in, total_out = sum(inside.values()), sum(g for g, *_ in outside)
    lines = [f"step {span:.1f} ms; idle {total_in + total_out:.1f} ms ({(total_in + total_out) / span:.1%}): "
             f"in-step {total_in:.1f} ms ({total_in / span:.1%}), out-step {total_out:.1f} ms ({total_out / span:.1%})",
             "\nin-step idle by step (ms)"]
    for kind, ms in inside.most_common(10):
        lines.append(f"  {ms:7.1f}  {kind}")
    lines.append("\nout-step gaps by size")
    for lo, hi in ((1.0, float("inf")), (0.1, 1.0), (0.0, 0.1)):
        sel = [g for g, *_ in outside if lo < g <= hi]
        lines.append(f"  {lo:g}-{hi:g} ms: {len(sel)} gaps, {sum(sel):.1f} ms")
    lines.append("\nlargest out-step gaps (ms, at ms, step before -> step after, collective that ended last)")
    for g, at, before, after, ending in sorted(outside, reverse=True)[:largest]:
        lines.append(f"  {g:6.2f} at {at:8.1f}  {before[-40:]} -> {after[-40:]}  [{ending[:40]}]")
    bins, lead = [0.0] * 20, [[] for _ in range(20)]
    for start, length, *_ in found:
        bins[min(19, int((start - t.t0) / (t.t1 - t.t0) * 20))] += length / 1e3
    for k in t.on_compute():
        launch = t.launch(k)
        if launch:
            lead[min(19, int((k["ts"] - t.t0) / (t.t1 - t.t0) * 20))].append((k["ts"] - launch[0]) / 1e3)
    lines.append("\nper twentieth of the step: idle ms / median host lead ms")
    lines.append("  " + " ".join(f"{b:.0f}/{sorted(v)[len(v) // 2] if v else 0:.0f}" for b, v in zip(bins, lead)))
    return "\n".join(lines)
