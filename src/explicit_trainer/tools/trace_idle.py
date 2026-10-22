"""Where the compute stream of a profiled step waits: its busy time, and every idle gap attributed to
the step that ended last on another stream before the compute stream resumed (what it waited for)
and to the step that resumed.

Purpose and design. Reads a PyTorch profiler trace of a run whose trainer annotated its steps
("step <kind> #<index>", Trainer.annotate). The compute stream is the stream with the most kernels
(as steptrace.py decides) unless given. Gaps under 0.2 ms are ignored. The report gives the busiest
streams with their most frequent step kinds, the compute stream's span and busy time, and the idle
time by cause and by the step that resumed. A trace of several hundred MB takes several GB of memory.

Public components
    idle(path, stream=None) -> dict with "streams", "span_s", "busy_s", "by_cause", "by_resumed"
    report(path, stream=None) -> str

Example
    print(report("trace_rank0.json.gz"))
"""
import bisect
import collections
import gzip
import json


def idle(path: str, stream=None) -> dict:
    ev = [e for e in json.load(gzip.open(path))["traceEvents"] if e.get("ph") == "X"]
    ker = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    ann = [e for e in ev if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("step ")]
    busy = collections.Counter()
    for e in ker:
        busy[e["tid"]] += e["dur"]
    kinds = collections.defaultdict(collections.Counter)
    for e in ann:
        kinds[e["tid"]][e["name"].split()[1].split(".")[0]] += 1
    if stream is None:
        stream = collections.Counter(e["tid"] for e in ker if e.get("cat") == "kernel").most_common(1)[0][0]
    ck = sorted((e for e in ker if e["tid"] == stream), key=lambda e: e["ts"])
    others = sorted((e for e in ann if e["tid"] != stream), key=lambda e: e["ts"] + e["dur"])
    ends = [e["ts"] + e["dur"] for e in others]
    own = sorted((e for e in ann if e["tid"] == stream), key=lambda e: e["ts"])
    own_starts = [e["ts"] for e in own]
    by_cause, by_resumed = collections.Counter(), collections.Counter()
    end = ck[0]["ts"] + ck[0]["dur"]
    for e in ck[1:]:
        gap = e["ts"] - end
        if gap > 200:
            j = bisect.bisect_right(ends, e["ts"]) - 1
            cause = others[j]["name"].split()[1] if j >= 0 and ends[j] > end else "(the host or the same stream)"
            a = bisect.bisect_right(own_starts, e["ts"] + 1) - 1
            by_cause[cause] += gap
            by_resumed[own[a]["name"].split()[1] if a >= 0 else "?"] += gap
        end = max(end, e["ts"] + e["dur"])
    return {"streams": {t: (busy[t] / 1e6, dict(kinds[t].most_common(3))) for t, _ in busy.most_common(10)},
            "span_s": (ck[-1]["ts"] + ck[-1]["dur"] - ck[0]["ts"]) / 1e6, "busy_s": sum(e["dur"] for e in ck) / 1e6,
            "by_cause": {k: v / 1e6 for k, v in by_cause.most_common(15)},
            "by_resumed": {k: v / 1e6 for k, v in by_resumed.most_common(10)}}



def report(path: str, stream=None) -> str:
    r = idle(path, stream)
    lines = [f"stream {t:>4} busy {b:8.2f} s  {k}" for t, (b, k) in r["streams"].items()]
    lines.append(f"compute stream: span {r['span_s']:.2f} s, busy {r['busy_s']:.2f} s")
    lines.append("idle by what ended last on another stream:")
    lines += [f"  {v:8.3f} s  {k}" for k, v in r["by_cause"].items()]
    lines.append("idle by the compute step that resumed:")
    lines += [f"  {v:8.3f} s  {k}" for k, v in r["by_resumed"].items()]
    return "\n".join(lines)
