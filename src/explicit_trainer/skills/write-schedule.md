---
name: write-schedule
description: Generate, write or change the schedule of a training step's graph in explicit-trainer (which stream each step runs on and in which order). Use after any change of the plan, or to overlap communication with compute.
---

# Writing a schedule

A schedule is data (`explicit_trainer.framework.schedule`): `{"graph": <hash>, "queues": {<stream>: [step index, ...]}}`.

- Every step of the graph is in exactly one queue. Streams: `compute`, `to_host` (small copies to the host),
  `host` (compute steps on host values, run on the CPU), or any other name: a CUDA stream of its own whose
  collectives get a process group of that name. By the step's type in `describe(graph)`: "input" goes on `compute`;
  "to_host" on `to_host`; "host" on `host`; "compute" on any stream but `to_host` and `host`; "collective" on any
  stream but `compute`, `to_host` and `host`.
- Every device runs the same schedule; collectives on one stream are matched across devices by their order there.
- The trainer launches the head of any queue whose dependencies are launched and whose host values have arrived,
  and waits only when no head can go. A step waits on the device for dependencies that ran on other device streams.

## Steps

1. Build the graph offline and describe it: `info = describe(plan.build())` (`explicit_trainer.framework.schedule`);
   write it as `graph.json`. Building a plan needs no GPU and no process group; when the model's code imports
   GPU-only packages, run the script with them stubbed: `et-stub --stub triton,<other packages> script.py ...`.
2. Write the schedule from `info["steps"]` (each has `kind`, `type`, `deps`, `micro_batch`): program order (every
   queue in increasing step index) is always valid when each type goes to an allowed stream;
   `examples/tiny/schedule.py` does exactly that. Then place steps for overlap: start gathers and exchanges early on
   their own streams, keep each micro-batch group's memory bounded.
3. Validate: `et-schedule validate graph.json schedule.json` (right graph, every step once, allowed streams, can be
   launched to the end); the trainer runs the same check and refuses a schedule whose graph hash differs. The hash
   covers every step's kind (file and function name), type and dependencies: regenerate the schedule after every
   change of the plan or of code that builds the graph.
4. Estimate before spending GPU time (skill `validate-and-simulate`).
