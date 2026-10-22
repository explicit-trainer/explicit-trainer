"""Where the GPU time of a training step goes, per kind of compute step.

Purpose and design. Every kernel of the compute stream is charged to the kind of step that launched
it (steptrace.Trace.step_of; "(outside steps)" for work the framework launches itself). For each
kind, in order of GPU time: the number of calls, the GPU time and its share of the compute stream's
busy time, the kernels per call, and the N kernels that take most of its time (time, share, count
per call). Given the FLOPs of every kind in one training step ({kind: [FP8 FLOPs, BF16 FLOPs]}, which
the model's own code computes) and the device's dense peaks, each kind also gets its MFU: its FLOPs at
those peaks over its GPU time. The report is plain text, for people and agents alike.

Public components
    kinds(trace) -> {kind: (calls, GPU s, kernels, {kernel name: (GPU s, count)})}
        The per-kind summary, from a steptrace.Trace, largest GPU time first.
    report(trace, flops=None, peaks=None, kernels=5) -> str
        The report. flops: {kind: [FP8 FLOPs, BF16 FLOPs]}; peaks: (FP8, BF16) dense FLOP/s of the
        device, required with flops.

Example
    t = Trace("trace_rank0.json.gz")
    print(report(t, json.load(open("flops.json")), peaks=(296e12, 148e12)))
"""
import collections

from explicit_trainer.tools.steptrace import Trace


def kinds(trace: Trace) -> dict:
    calls = collections.Counter(s[3] for s in trace.steps)
    time, count = collections.Counter(), collections.Counter()
    per = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))
    for k in trace.on_compute():
        i = trace.step_of(k)
        kind = trace.steps[i][3] if i >= 0 else "(outside steps)"
        time[kind] += k["dur"] / 1e6
        count[kind] += 1
        per[kind][k["name"]][0] += k["dur"] / 1e6
        per[kind][k["name"]][1] += 1
    return {kind: (calls[kind] or 1, s, count[kind], dict(per[kind])) for kind, s in time.most_common()}


def report(trace: Trace, flops: dict | None = None, peaks: tuple[float, float] | None = None, kernels: int = 5) -> str:
    assert flops is None or peaks is not None, "FLOPs need the device's peaks"
    summary = kinds(trace)
    busy = sum(s for _, s, _, _ in summary.values())
    ideal_total = 0.0
    lines = [f"compute stream busy {busy:.3f} s over {(trace.t1 - trace.t0) / 1e6:.3f} s",
             f"{'step':52s} {'calls':>5s} {'GPU s':>7s} {'share':>6s} {'kern/call':>9s} {'MFU':>6s}"]
    for kind, (calls, s, n, _) in summary.items():
        mfu = "     -"
        if flops is not None and kind in flops:
            ideal = flops[kind][0] / peaks[0] + flops[kind][1] / peaks[1]
            ideal_total += ideal
            mfu = f"{ideal / s:6.1%}"
        lines.append(f"{kind[-52:]:52s} {calls:5d} {s:7.3f} {s / busy:6.1%} {n / calls:9.1f} {mfu}")
    if flops is not None:
        lines.append(f"{'all compute steps':52s} {'':5s} {busy:7.3f} {'':6s} {'':9s} {ideal_total / busy:6.1%}")
    lines.append("\nlongest kernels per step (GPU s, share of the step, per call)")
    for kind, (calls, s, _, per) in summary.items():
        if s < 0.01 * busy:
            break
        lines.append(f"{kind}:")
        for name, (t, n) in sorted(per.items(), key=lambda kv: -kv[1][0])[:kernels]:
            lines.append(f"  {t:7.3f} {t / s:6.1%} {n / calls:5.1f}  {name[:90]}")
    return "\n".join(lines)
