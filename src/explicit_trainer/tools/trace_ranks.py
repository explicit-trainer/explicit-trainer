"""Compares the ranks of one profiled step: which rank arrives last at a synchronizing step kind (an
exchange all ranks must reach), how late, and whether its host launched it late, from the traces of
several ranks.

Purpose and design. Two stages, so that the large traces are read where they are and only small
records travel. extract: for one trace, the host launch times and the GPU start times and durations
of the steps whose kind contains a substring, in order (one JSON record per trace). compare: per
call, the spread of the GPU start times over the ranks; how often each rank was last; the calls
spread over more than 100 ms with every rank's GPU start and host launch offset. Ranks are read
from the trace file names (trace_rank<N>). Host clocks of different machines are compared as they
are.

Public components
    extract(path, kind) -> {"trace", "host": [launch times], "gpu": [[start, duration]]}
    compare(records) -> dict with "last" (count per rank), "spread_ms", "late_calls"
    report(records) -> str

Example
    records = [extract(f"trace_rank{r}.json.gz", "_combine") for r in range(4)]
    print(report(records))
"""
import gzip
import json
import re
import statistics


def extract(path: str, kind: str) -> dict:
    ev = [e for e in json.load(gzip.open(path))["traceEvents"]
          if e.get("ph") == "X" and e.get("name", "").startswith("step ") and kind in e["name"]]
    return {"trace": path, "host": sorted(e["ts"] for e in ev if e.get("cat") == "user_annotation"),
            "gpu": sorted([e["ts"], e["dur"]] for e in ev if e.get("cat") == "gpu_user_annotation")}


def compare(records: list) -> dict:
    by_rank = {int(re.search(r"rank(\d+)", r["trace"]).group(1)): r for r in records}
    ranks = sorted(by_rank)
    n = min(len(by_rank[r]["gpu"]) for r in ranks)
    last, spread, late = {r: 0 for r in ranks}, [], []
    for k in range(n):
        starts = {r: by_rank[r]["gpu"][k][0] for r in ranks}
        last[max(starts, key=starts.get)] += 1
        spread.append((max(starts.values()) - min(starts.values())) / 1e3)
        if spread[-1] > 100:
            g0 = min(starts.values())
            host = {r: by_rank[r]["host"][k] for r in ranks if len(by_rank[r]["host"]) > k}
            h0 = min(host.values()) if host else 0
            late.append({"call": k, "gpu_ms": {r: round((starts[r] - g0) / 1e3) for r in ranks},
                         "host_ms": {r: round((t - h0) / 1e3) for r, t in host.items()}})
    return {"last": last, "spread_ms": {"median": statistics.median(spread), "max": max(spread), "total": sum(spread)},
            "late_calls": late}



def report(records: list) -> str:
    r = compare(records)
    lines = [f"last to start, per rank: {r['last']}",
             "spread of GPU starts per call (ms): median %.1f, max %.1f, total %.0f" % tuple(r["spread_ms"].values()),
             f"calls spread over 100 ms: {len(r['late_calls'])}"]
    for c in r["late_calls"][:8]:
        lines.append(f"call {c['call']} GPU start offsets {c['gpu_ms']}")
        lines.append(f"        host launch offsets {c['host_ms']}")
    return "\n".join(lines)
