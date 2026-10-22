"""et-trace: reports on profiled training steps (PyTorch profiler traces with step annotations).

et-trace steps <trace> [--flops flops.json --peaks FP8 BF16] [--kernels N]   GPU time and MFU per kind of step
et-trace gaps <trace> [--largest N]                                          idle time of the compute stream
et-trace timeline <trace> [MS ...] [--span MS] [--png out.png --from MS --to MS]
et-trace idle <trace> [--stream TID]                                         idle time by cause
et-trace diff <trace A> <trace B> [--rows N]                                 GPU time per kind, A against B
et-trace ranks-extract <kind substring> <trace>...                           JSON lines, one per trace
et-trace ranks-compare <calls.jsonl>...                                      which rank is last, how late

Each subcommand prints the report of its module in explicit_trainer.tools (steps, gaps, timeline,
trace_idle, trace_diff, trace_ranks).

Public components
    main()

Example
    et-trace steps trace_rank0.json.gz --flops flops.json --peaks 296e12 148e12
    et-trace timeline trace_rank0.json.gz 2434.2 --span 3
"""
import argparse
import json

from explicit_trainer.tools import gaps, steps, timeline, trace_diff, trace_idle, trace_ranks
from explicit_trainer.tools.steptrace import Trace


def main():
    p = argparse.ArgumentParser(prog="et-trace")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("steps")
    s.add_argument("trace")
    s.add_argument("--flops")
    s.add_argument("--peaks", type=float, nargs=2, metavar=("FP8", "BF16"), help="dense FLOP/s of the device")
    s.add_argument("--kernels", type=int, default=5)
    s = sub.add_parser("gaps")
    s.add_argument("trace")
    s.add_argument("--largest", type=int, default=15)
    s = sub.add_parser("timeline")
    s.add_argument("trace")
    s.add_argument("times", type=float, nargs="*")
    s.add_argument("--span", type=float, default=5.0)
    s.add_argument("--png")
    s.add_argument("--from", dest="start", type=float, default=0.0)
    s.add_argument("--to", dest="end", type=float)
    s = sub.add_parser("idle")
    s.add_argument("trace")
    s.add_argument("--stream", type=int)
    s = sub.add_parser("diff")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--rows", type=int, default=20)
    s = sub.add_parser("ranks-extract")
    s.add_argument("kind")
    s.add_argument("traces", nargs="+")
    s = sub.add_parser("ranks-compare")
    s.add_argument("calls", nargs="+")
    a = p.parse_args()
    if a.command == "steps":
        flops = json.load(open(a.flops)) if a.flops else None
        print(steps.report(Trace(a.trace), flops, tuple(a.peaks) if a.peaks else None, a.kernels))
    elif a.command == "gaps":
        print(gaps.report(Trace(a.trace), a.largest))
    elif a.command == "timeline":
        t = Trace(a.trace)
        if a.png:
            timeline.gantt(t, a.png, a.start, a.end if a.end is not None else (t.t1 - t.t0) / 1e3)
        for ms in a.times:
            print("\n".join(timeline.around(t, ms, a.span)))
    elif a.command == "idle":
        print(trace_idle.report(a.trace, a.stream))
    elif a.command == "diff":
        print(trace_diff.report(a.a, a.b, a.rows))
    elif a.command == "ranks-extract":
        for path in a.traces:
            print(json.dumps(trace_ranks.extract(path, a.kind)), flush=True)
    else:
        print(trace_ranks.report([json.loads(line) for path in a.calls for line in open(path) if line.strip()]))


if __name__ == "__main__":
    main()
