---
name: validate-and-simulate
description: Check a schedule against its graph and estimate the step time under it offline, with step times from a profile; search a better order. Use before running a new schedule on GPUs.
---

# Validating and simulating schedules

All of this runs without GPUs, from `graph.json` (`describe(graph)`) and one profiled step of the same graph (any
schedule of it: step times are taken per step index).

- `et-schedule validate graph.json a.json [b.json ...]`
- `et-schedule estimate graph.json a.json [b.json ...] --profile trace_rank0.json.gz [--gaps 10]
  [--around MS ... --span MS] [--png sim.png --from MS --to MS]`
  prints per schedule: the simulated step (and the profile's measured one), busy and idle time per stream, idle
  time by what the next step waited for, the largest gaps of the compute stream, the critical path by kind and
  stream. Compare the simulation of the profiled schedule with its measured step first: that is the simulator's error.
- `et-order graph.json schedule.json --profile trace_rank0.json.gz --group N --minutes M --out searched.json`
  searches an order of the same streams (critical-path start, then random moves within the dependencies).

Not modelled: other ranks' skew beyond what the profiled collectives include, compute slowed by concurrent
collectives, the CUDA launch queue limit, memory. A schedule that wins in simulation must still be measured.
