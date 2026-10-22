"""What holds device memory at the peak of a recorded step: the live allocations at the moment the
step's allocated memory was highest, summed by the code location that made them, and the largest ones
in allocation order.

Purpose and design. Reads a PyTorch memory snapshot (torch.cuda.memory._dump_snapshot of a step
recorded with torch.cuda.memory._record_memory_history). Only allocations made while recording are
seen: the persistent state allocated before the step is not in the peak (the snapshot's segments,
also reported, include it). A location is the first frames outside site-packages; allocations made
inside libraries without Python frames (autograd, NCCL) show their C++ frames instead.

Public components
    peak(path) -> (peak bytes, event index, {address: (event index, event)} live at the peak, segment bytes)
    where(event, frames=4) -> str
        The code location of an allocation.
    report(path, rows=20) -> str

Example
    print(report("memory_rank0.pickle", 25))
"""
import collections
import pickle


def peak(path: str):
    snap = pickle.load(open(path, "rb"))
    live, cur, best, best_live, at = {}, 0, 0, {}, 0
    for i, e in enumerate(max(snap["device_traces"], key=len)):  # the trace of the rank's own device
        if e["action"] == "alloc":
            live[e["addr"]] = (i, e)
            cur += e["size"]
            if cur > best:
                best, best_live, at = cur, dict(live), i
        elif e["action"] == "free_completed" and e["addr"] in live:
            cur -= live.pop(e["addr"])[1]["size"]
    return best, at, best_live, sum(s["total_size"] for s in snap["segments"])


def where(e: dict, frames: int = 4) -> str:
    fr = [f for f in e.get("frames", []) if f["filename"].endswith(".py") and "site-packages" not in f["filename"]
          and "/lib/python" not in f["filename"]]
    if not fr:
        return "C++: " + " < ".join(f["name"][:40] for f in e.get("frames", [])[3:6])
    return " < ".join(f"{f['name']}:{f['filename'].split('/')[-1]}:{f['line']}" for f in fr[:frames])



def report(path: str, rows: int = 20) -> str:
    best, at, live, segments = peak(path)
    lines = [f"peak of the step's allocations {best / 2**30:.2f} GiB at event {at}; "
             f"segments {segments / 2**30:.2f} GiB"]
    size, count = collections.Counter(), collections.Counter()
    for _, e in live.values():
        size[where(e)] += e["size"]
        count[where(e)] += 1
    for k, v in size.most_common(rows):
        lines.append(f"{v / 2**30:7.2f} GiB {count[k]:5d}  {k[:200]}")
    lines.append("largest live allocations, in allocation order:")
    for i, e in sorted(v for v in live.values() if v[1]["size"] >= 2**30):
        lines.append(f"  event {i:8d} {e['size'] / 2**30:6.2f} GiB  {where(e, 1)[:120]}")
    return "\n".join(lines)
