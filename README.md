# explicit-trainer

A deep-learning training framework in which a training step is written out explicitly, built for agents that optimize
it: a model of modules with declared tensors, a placement of every tensor on the device mesh, a plan that builds one
training step as a static graph of explicit steps (compute, collectives, copies), and a schedule file that places the
steps on streams. The trainer launches the graph as the schedule says. Every step can be checked against a plain
PyTorch reference, schedules are generated, validated and simulated without GPUs, and every step is named in profiler
traces.

## Install

```
pip install -e .
python -m unittest discover -s tests -t .
torchrun --nproc-per-node 2 -m examples.tiny.run
```

## Layout

| Path | Content |
|---|---|
| `src/explicit_trainer/framework/` | `abstract/` (modules, placements, steps and graphs, module implementations, plans), `schedule.py`, `trainer.py` (`docs/framework.md`) |
| `src/explicit_trainer/tools/` | offline analysis: schedules (validate, simulate, search an order), profiled steps (time per step, idle time, timelines, dependencies, ranks), memory snapshots, running code with GPU-only packages stubbed |
| `src/explicit_trainer/skills/` | instructions for agents, one Markdown file per skill (`et-skills install .claude/skills`) |
| `src/explicit_trainer/loop/` | the scripted side of an agent's run toward a goal: `abstract/` (GpuProvider, Checker, Hook), their implementations in `gpu_provider/`, `checker/` and `hook/`, `loop.py` (Loop), `server.py` (a UNIX socket daemon and its client) |
| `src/explicit_trainer/cli/` | the commands below |
| `examples/tiny/` | a complete small model, placement, implementations, plan and schedule, trained and checked on CPU with gloo |
| `tests/` | unittest tests, no GPU needed |
| `docs/` | `framework.md` (the design), `workflow.md` (the loop of a change) |
| `AGENTS.md` | rules for AI agents working here |

## Commands

| Command | What |
|---|---|
| `et-schedule validate\|estimate` | check schedules against their graph; simulate a step under them with profiled step times |
| `et-order` | search a schedule's order on simulated steps |
| `et-deps` | the steps of a time range of a profiled step with their dependencies |
| `et-trace steps\|gaps\|timeline\|idle\|diff\|ranks-extract\|ranks-compare` | reports on profiled steps |
| `et-memsnap` | what holds device memory at the peak of a recorded step |
| `et-stub` | run a script with GPU-only packages replaced by stubs |
| `et-skills list\|install` | the skills |
| `et-loop start\|msg\|resume\|state` | talk to a loop served on a UNIX socket |

Each module's header documents its interface in full.
