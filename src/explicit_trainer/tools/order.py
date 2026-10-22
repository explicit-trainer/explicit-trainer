"""Searching the order of a schedule on simulated steps.

Purpose and design. Starting from a given schedule of a graph, with the time of every step taken from
a profiled step of the same graph (simulate.durations), the search simulates steps (simulate.py)
under other orders and keeps the best. It starts from the better of the given schedule and a
critical-path order of it (micro-batch groups kept in sequence for memory; inside each, the step
with the longest remaining host and device time first), then tries random moves of a few steps
within what their dependencies allow, evaluated on all cores, keeping improvements, for a given
time. An order is a list of all step indices; it becomes a schedule by the given schedule's streams
(each queue takes its steps in that order), which the search never changes.

Public components
    merged(info, queues) -> [step]
        One order of all steps that keeps each queue's order and the dependencies.
    critical_path(info, order, host, device, group) -> [step]
        group: micro-batches per group (micro-batch m is in group m // group).
    schedule_of(info, streams, order) -> dict
        streams: {step: stream}; the schedule of an order.
    search(info, schedule, host, device, group, minutes, log=print) -> (schedule, {name: simulated s})
        The best schedule found, and the simulated step of the given schedule ("given"), of its
        critical-path order ("critical path") and of the best ("best"). log receives progress lines.

Example
    info, given = json.load(open("graph.json")), json.load(open("schedule.json"))
    host, device = durations(Trace("trace_rank0.json.gz"), len(info["steps"]))
    best, scores = search(info, given, host, device, group=2, minutes=20)
"""
import heapq
import multiprocessing
import os
import time

import numpy as np

from explicit_trainer.tools.simulate import simulate

_ctx = {}  # what a worker process needs: info, streams, host, device, users


def merged(info: dict, queues: dict) -> list[int]:
    steps = info["steps"]
    heads, placed, out = {q: 0 for q in queues}, [False] * len(steps), []
    while len(out) < len(steps):
        for q, members in queues.items():
            while heads[q] < len(members) and all(placed[d] for d in steps[members[heads[q]]]["deps"]):
                placed[members[heads[q]]] = True
                out.append(members[heads[q]])
                heads[q] += 1
    return out


def _users(info: dict) -> list[list[int]]:
    users = [[] for _ in info["steps"]]
    for i, s in enumerate(info["steps"]):
        for d in s["deps"]:
            users[d].append(i)
    return users


def critical_path(info: dict, order: list[int], host: list[float], device: list[float], group: int) -> list[int]:
    steps, users = info["steps"], _users(info)
    bottom = [0.0] * len(steps)
    for i in reversed(order):
        bottom[i] = host[i] + device[i] + max((bottom[u] for u in users[i]), default=0.0)
    position = {i: k for k, i in enumerate(order)}
    groups, seen = [None] * len(steps), {}
    for i in order:  # a step without a micro-batch joins the group of the step before it
        mb = steps[i]["micro_batch"]
        g = mb // group if mb is not None else (groups[order[position[i] - 1]] if position[i] else -1)
        groups[i] = seen.setdefault(g, len(seen))
    waiting = [len(s["deps"]) for s in steps]
    ready = [(groups[i], -bottom[i], i) for i in range(len(steps)) if not waiting[i]]
    heapq.heapify(ready)
    out = []
    while ready:
        i = heapq.heappop(ready)[2]
        out.append(i)
        for u in users[i]:
            waiting[u] -= 1
            if not waiting[u]:
                heapq.heappush(ready, (groups[u], -bottom[u], u))
    return out


def schedule_of(info: dict, streams: dict, order: list[int]) -> dict:
    queues = {}
    for i in order:
        queues.setdefault(streams[i], []).append(i)
    return dict(graph=info["hash"], queues=queues)


def _simulated(order: list[int]) -> float:
    return simulate(_ctx["info"], schedule_of(_ctx["info"], _ctx["streams"], order), _ctx["host"],
                    _ctx["device"])["step_s"]


def _moved(order: list[int], rng) -> list[int]:
    """The order with one step moved to another place its dependencies allow."""
    steps, users = _ctx["info"]["steps"], _ctx["users"]
    i = order[int(rng.integers(len(order)))]
    out = [s for s in order if s != i]
    pos = {s: k for k, s in enumerate(out)}
    lo = max((pos[d] for d in steps[i]["deps"]), default=-1) + 1
    hi = min((pos[u] for u in users[i]), default=len(out))
    out.insert(int(rng.integers(lo, hi + 1)), i)
    return out


def _try(job: tuple[list[int], int]) -> tuple[float, list[int]]:
    """A few random moves from `order`, and the simulated step of the result."""
    order, seed = job
    rng = np.random.default_rng(seed)
    o = _moved(order, rng)
    for _ in range(int(rng.integers(1, 8))):
        o = _moved(o, rng)
    return _simulated(o), o


def _init(ctx: dict):
    _ctx.update(ctx)


def search(info: dict, schedule: dict, host: list[float], device: list[float], group: int, minutes: float,
           log=print) -> tuple[dict, dict]:
    streams = {i: q for q, members in schedule["queues"].items() for i in members}
    ctx = dict(info=info, streams=streams, host=host, device=device, users=_users(info))
    _init(ctx)
    start_order = merged(info, schedule["queues"])
    candidates = {"given": start_order, "critical path": critical_path(info, start_order, host, device, group)}
    scores = {name: _simulated(o) for name, o in candidates.items()}
    for name, t in scores.items():
        log(f"{name}: simulated {t:.3f} s")
    best = candidates[min(scores, key=scores.get)]
    best_t, seed, start = min(scores.values()), 0, time.time()
    batch = 4 * os.cpu_count()
    with multiprocessing.get_context("fork").Pool(initializer=_init, initargs=(ctx,)) as pool:
        while time.time() - start < minutes * 60:
            t, o = min(pool.map(_try, [(best, s) for s in range(seed, seed + batch)]), key=lambda r: r[0])
            seed += batch
            if t < best_t:
                best, best_t = o, t
                log(f"{time.time() - start:6.0f} s: simulated {best_t:.3f} s")
    return schedule_of(info, streams, best), scores | {"best": best_t}
