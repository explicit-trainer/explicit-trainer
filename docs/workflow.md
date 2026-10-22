# Workflow

A change to the training step goes through the same loop every time.

## 1. Understand

- `docs/framework.md`: modules, placements, plans (a training step as one static graph of explicit steps),
  schedules (one queue of steps per stream, a file), the trainer.
- The model's own code: its plan, its schedule generator, the configurations it runs and what was measured.
- `examples/tiny/` shows every piece in its smallest form.

## 2. Edit, then build the step's graph offline

Every configuration needs its graph and schedule, built from the code without GPUs: a script that builds the plan,
writes `describe(plan.build())` as `graph.json` and the generated schedule next to it. On a machine without GPUs, stub
the packages that need them:

```
et-stub --stub triton,<other GPU-only packages> build_schedule.py config.json
et-schedule validate graph.json schedule.json
```

Regenerate after every change of the plan or the code that builds the graph: the trainer refuses a schedule whose
graph hash differs.

## 3. Check correctness

`Trainer.check(...)` runs one training step next to the model's references (on a reduced model when the whole one does
not fit on one device in FP32) and reports both losses, every state gradient's relative error and, with `leaves=True`,
the errors of every implemented call. Run it after any change of the step and compare with the levels documented for
the model.

## 4. Run on the GPUs

Check that the GPUs are free first. The first step of a run executes serially (`step(serial=True)`: kernels load on an
idle device) and is not timed. Record per step the loss, the time and the memory (`torch.cuda.max_memory_allocated`,
`torch.cuda.memory_reserved`).

## 5. Find where time and memory go

- Profile one step with `trainer.annotate = True` and export each rank's trace (`trace_rank<N>.json.gz`):
  `et-trace steps|gaps|timeline|idle|diff|ranks-*` and `et-deps` read them (skill `read-trace`).
- Simulate other schedules with the profile before running them: `et-schedule estimate`, `et-order`
  (skill `validate-and-simulate`).
- Record every allocation of one step (`torch.cuda.memory._record_memory_history`, then `_dump_snapshot`):
  `et-memsnap` shows what holds memory at the peak.

Then change one thing and go back to 2 (skill `profile-and-attack`).
