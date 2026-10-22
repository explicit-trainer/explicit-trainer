"""Schedules: which stream each step of a graph runs on, and in which order on that stream.

Purpose and design. A schedule is data, written once per graph by a generator (a plan's own code)
or by hand, and read by the trainer. It has one
queue per stream: the steps of that stream in the order they are launched there. As a JSON file:

    {"graph": "<hash of the graph>", "queues": {"<stream>": [step index, ...], ...}}

Every step is in exactly one queue. A stream is "compute" (the device's compute stream),
"to_host" (copies of small tensors to the host), "host" (compute steps on host values, run on the
CPU), or any other name: a CUDA stream of its own, whose collectives run on a process group of
that name per set of mesh dimensions. Device compute steps may run on any stream but to_host and
host (e.g. a step that communicates by itself). Every device runs the same schedule; collectives on
one stream are matched across devices by their order there, which the schedule fixes.

What the trainer does with it (and explicit_trainer.tools.simulate simulates): it launches the head of any queue
whose dependencies are all launched and whose host values (outputs of to_host steps) have arrived,
and waits only when no head can be launched. Only the interleaving of the queues depends on
timing. A step waits on the device for the steps it depends on that ran on other device streams.

Public components
    kind(step) -> str
        "<file>.<function>" of a compute step, else the step's type (e.g. AllToAll).
    dependencies(graph) -> [[int]]
        Per step, the indices of the steps it waits for: makers of the Values it reads, and earlier
        accesses to the same copy (a read follows the writes before it; a write follows the reads
        and writes before it; a step's reads come before its writes).
    describe(graph) -> dict
        The graph as offline tools need it: "hash", and per step "kind", "type" (input, compute,
        host, to_host or collective), "deps" and "micro_batch" (the one micro-batch its inputs and
        copies belong to, else None).
    check(schedule, info)
        info: describe(graph). Checks that the schedule is for this graph, has every step in one
        queue of a stream its type allows, and can be launched to the end (no queue waits forever
        for a step behind the head of another); fails loudly otherwise.

Example
    info = describe(graph)
    queues = {}
    for i, s in enumerate(info["steps"]):  # in program order, collectives on one stream
        q = {"input": "compute", "collective": "comm"}.get(s["type"], s["type"])
        queues.setdefault(q, []).append(i)
    json.dump({"graph": info["hash"], "queues": queues}, open("schedule.json", "w"))
    trainer = Trainer(plan, json.load(open("schedule.json")), mesh)
"""
import hashlib
import json

from explicit_trainer.framework.abstract.step import CommunicationStep, ComputeStep, Graph, Input, Step, ToHost

DEVICE_ONLY = ("to_host", "host")


def kind(s: Step) -> str:
    return f"{s.fn.__module__.rsplit('.', 1)[-1]}.{s.fn.__qualname__}" if isinstance(s, ComputeStep) \
        else type(s).__name__


def dependencies(graph: Graph) -> list[list[int]]:
    index = {s: i for i, s in enumerate(graph.steps)}
    out, last_write, reads = [], {}, {}
    for s in graph.steps:
        d = {index[v.step]: None for v in s.values()}
        for c in s.reads():
            if c.key in last_write:
                d[last_write[c.key]] = None
            reads.setdefault(c.key, []).append(index[s])
        for w in s.writes:
            k = w.copy.key
            if k in last_write:
                d[last_write[k]] = None
            d.update(dict.fromkeys(r for r in reads.pop(k, []) if r != index[s]))
            last_write[k] = index[s]
        d.pop(index[s], None)
        out.append(list(d))
    return out


def describe(graph: Graph) -> dict:
    index = {s: i for i, s in enumerate(graph.steps)}
    types, micro = [], []
    for s in graph.steps:
        made = [index[v.step] for v in s.values()]
        if isinstance(s, Input):
            types.append("input")
        elif isinstance(s, ToHost):
            types.append("to_host")
        elif isinstance(s, CommunicationStep):
            types.append("collective")
        elif made and all(types[i] in DEVICE_ONLY for i in made):
            types.append("host")
        else:
            types.append("compute")
        found = {s.micro_batch} if isinstance(s, Input) else \
            {micro[i] for i in made} | {c.micro_batch for c in s.reads()} | {w.copy.micro_batch for w in s.writes}
        found.discard(None)
        micro.append(found.pop() if len(found) == 1 else None)
    steps = [dict(kind=kind(s), type=t, deps=d, micro_batch=m)
             for s, t, d, m in zip(graph.steps, types, dependencies(graph), micro)]
    digest = hashlib.sha1(json.dumps([[s["kind"], s["type"], s["deps"]] for s in steps]).encode()).hexdigest()
    return dict(hash=digest, steps=steps)


def check(schedule: dict, info: dict):
    assert schedule["graph"] == info["hash"], "the schedule is for another graph"
    steps, queues = info["steps"], schedule["queues"]
    assert sorted(i for q in queues.values() for i in q) == list(range(len(steps))), \
        "every step must be in exactly one queue"
    for q, members in queues.items():
        for i in members:
            t = steps[i]["type"]
            allowed = {"input": q == "compute", "to_host": q == "to_host", "host": q == "host",
                       "compute": q not in DEVICE_ONLY, "collective": q not in DEVICE_ONLY + ("compute",)}
            assert allowed[t], ("step on a stream its type does not allow", i, t, q)
    heads, launched = {q: 0 for q in queues}, [False] * len(steps)
    moved = True
    while moved:
        moved = False
        for q, members in queues.items():
            while heads[q] < len(members) and all(launched[d] for d in steps[members[heads[q]]]["deps"]):
                launched[members[heads[q]]] = True
                heads[q] += 1
                moved = True
    stuck = {q: members[heads[q]] for q, members in queues.items() if heads[q] < len(members)}
    assert not stuck, ("these heads wait for steps behind other heads", stuck)
