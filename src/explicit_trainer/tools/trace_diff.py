"""The GPU time of every step kind in two profiled steps, side by side, largest change first: what a
change of configuration or code made slower or faster.

Purpose and design. Sums the duration of the trainer's step annotations ("step <kind> #<index>") per
kind in each trace (a step's GPU span on its stream, including waits inside its kernels, e.g. a
collective waiting for its peers). The report gives the busiest stream's kernel time of both traces,
then per kind: seconds in A, in B, the change, and the counts.

Public components
    kinds(path) -> (seconds per kind, count per kind, busiest stream's kernel seconds)
    report(path_a, path_b, rows=20) -> str

Example
    print(report("a/trace_rank0.json.gz", "b/trace_rank0.json.gz", 25))
"""
import collections
import gzip
import json


def kinds(path: str):
    t, n, busy = collections.Counter(), collections.Counter(), collections.Counter()
    for e in json.load(gzip.open(path))["traceEvents"]:
        if e.get("ph") != "X":
            continue
        if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("step "):
            k = e["name"].split()[1]
            t[k] += e["dur"] / 1e6
            n[k] += 1
        elif e.get("cat") == "kernel":
            busy[e["tid"]] += e["dur"] / 1e6
    return t, n, max(busy.values())



def report(path_a: str, path_b: str, rows: int = 20) -> str:
    (a, na, ba), (b, nb, bb) = kinds(path_a), kinds(path_b)
    lines = [f"busiest stream: {ba:.2f} s vs {bb:.2f} s"]
    for k in sorted(set(a) | set(b), key=lambda k: -abs(b[k] - a[k]))[:rows]:
        lines.append(f"{a[k]:9.2f} {b[k]:9.2f} {b[k] - a[k]:+9.2f} s  {na[k]:6d} {nb[k]:6d}  {k}")
    return "\n".join(lines)
