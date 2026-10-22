"""A schedule of the example's graph: every step in program order, collectives on their own stream.

Purpose and design. Program order is always a valid order; each step goes to a stream its type
allows (inputs and compute steps on "compute", collectives on "comm", to_host copies on "to_host",
steps on host values on "host").

Public components
    generate(graph) -> dict
        The schedule (framework/schedule.py).

Example
    trainer = Trainer(plan, generate(plan.build()), mesh)
"""
from explicit_trainer.framework.schedule import describe

_STREAM = {"input": "compute", "compute": "compute", "collective": "comm", "to_host": "to_host", "host": "host"}


def generate(graph) -> dict:
    info = describe(graph)
    queues = {}
    for i, s in enumerate(info["steps"]):
        queues.setdefault(_STREAM[s["type"]], []).append(i)
    return {"graph": info["hash"], "queues": queues}
