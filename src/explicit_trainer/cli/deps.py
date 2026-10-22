"""et-deps: the steps of a time range of a profiled step with their dependencies, and what each waited for.

et-deps <graph.json> <trace.json.gz> --from MS --to MS [--png deps.png]

Prints one line per step (tools/deps.py) and, with --png, draws the chart.

Public components
    main()

Example
    et-deps graph.json trace_rank0.json.gz --from 1670 --to 1700 --png deps.png
"""
import argparse
import json

from explicit_trainer.tools.deps import draw, text
from explicit_trainer.tools.steptrace import Trace


def main():
    p = argparse.ArgumentParser(prog="et-deps")
    p.add_argument("graph")
    p.add_argument("trace")
    p.add_argument("--from", dest="start", type=float, required=True)
    p.add_argument("--to", dest="end", type=float, required=True)
    p.add_argument("--png")
    a = p.parse_args()
    info, t = json.load(open(a.graph)), Trace(a.trace)
    print(text(info, t, a.start, a.end))
    if a.png:
        draw(info, t, a.start, a.end, a.png)


if __name__ == "__main__":
    main()
