"""et-order: search the order of a schedule on simulated steps, and write the best.

et-order <graph.json> <schedule.json> --profile <trace.json.gz> --group N --out <searched.json> [--minutes 20]

group: micro-batches run together (their forwards, then their backwards), kept in sequence by the
critical-path start. See tools/order.py.

Public components
    main()

Example
    et-order graph.json schedule.json --profile trace_rank0.json.gz --group 2 --out searched.json --minutes 10
"""
import argparse
import json

from explicit_trainer.tools.order import search
from explicit_trainer.tools.simulate import durations
from explicit_trainer.tools.steptrace import Trace


def main():
    p = argparse.ArgumentParser(prog="et-order")
    p.add_argument("graph")
    p.add_argument("schedule")
    p.add_argument("--profile", required=True)
    p.add_argument("--group", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--minutes", type=float, default=20.0)
    a = p.parse_args()
    info, given = json.load(open(a.graph)), json.load(open(a.schedule))
    trace = Trace(a.profile)
    host, device = durations(trace, len(info["steps"]))
    print(f"measured {(trace.t1 - trace.t0) / 1e6:.3f} s", flush=True)
    best, scores = search(info, given, host, device, a.group, a.minutes, log=lambda s: print(s, flush=True))
    json.dump(best, open(a.out, "w"))
    print(f"wrote {a.out}: simulated {scores['best']:.3f} s (given schedule {scores['given']:.3f} s)")


if __name__ == "__main__":
    main()
