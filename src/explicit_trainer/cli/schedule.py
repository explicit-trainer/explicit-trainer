"""et-schedule: check schedules against their graph, and estimate a training step under them offline.

et-schedule validate <graph.json> <schedule.json> [<schedule.json> ...]
et-schedule estimate <graph.json> <schedule.json> [...] --profile <trace.json.gz> [--gaps N]
    [--around MS [MS ...] --span MS] [--png out.png --from MS --to MS]

graph.json is framework.schedule.describe(graph). estimate prints the profile's measured step and,
per schedule, the report of tools/simulate.py; --around lists the simulated steps near given times;
--png draws the first schedule's simulated step.

Public components
    main()

Example
    et-schedule estimate graph.json schedule.json other.json --profile trace_rank0.json.gz --png sim.png
"""
import argparse
import json

from explicit_trainer.framework.schedule import check
from explicit_trainer.tools.simulate import around, draw, durations, report, simulate
from explicit_trainer.tools.steptrace import Trace


def main():
    p = argparse.ArgumentParser(prog="et-schedule")
    p.add_argument("command", choices=["validate", "estimate"])
    p.add_argument("graph")
    p.add_argument("schedules", nargs="+")
    p.add_argument("--profile")
    p.add_argument("--gaps", type=int, default=10)
    p.add_argument("--around", type=float, nargs="*", default=[])
    p.add_argument("--span", type=float, default=1.0)
    p.add_argument("--png")
    p.add_argument("--from", dest="lo", type=float, default=0.0)
    p.add_argument("--to", dest="hi", type=float)
    a = p.parse_args()
    info = json.load(open(a.graph))
    schedules = {path: json.load(open(path)) for path in a.schedules}
    for path, schedule in schedules.items():
        check(schedule, info)
        print(f"{path}: valid for this graph")
    if a.command == "validate":
        return
    assert a.profile, "estimate needs --profile"
    trace = Trace(a.profile)
    host, device = durations(trace, len(info["steps"]))
    print(f"profile: measured step {(trace.t1 - trace.t0) / 1e3:.1f} ms, host {sum(host) * 1e3:.1f} ms, "
          f"device {sum(device) * 1e3:.1f} ms in steps")
    for k, (path, schedule) in enumerate(schedules.items()):
        sim = simulate(info, schedule, host, device)
        print(f"\n{path}\n{report(info, sim, a.gaps)}")
        for ms in a.around:
            print(around(info, sim, ms, a.span))
        if a.png and k == 0:
            draw(info, sim, a.png, a.lo, a.hi if a.hi is not None else sim["step_s"] * 1e3)


if __name__ == "__main__":
    main()
