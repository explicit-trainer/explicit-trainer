---
name: read-trace
description: Read a profiled training step of explicit-trainer (PyTorch profiler trace with step annotations) to find where GPU time and idle time go. Use after profiling a step.
---

# Reading a trace

Profile one step of a GPU run: set `trainer.annotate = True`, run the step inside
`torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])`, wait for the device, and export
each rank's trace with `prof.export_chrome_trace(f"trace_rank{rank}.json.gz")`. Every step appears as
"step <kind> #<index>": kind is `<file>.<function>` of a compute step or the collective's type (e.g. `AllReduce`);
index is its position in the graph.

- `et-trace steps T [--flops flops.json --peaks FP8 BF16] [--kernels N]`: GPU time per kind of step, its share,
  kernels per call, the heaviest kernels; with FLOPs per kind and the device's dense peaks, the MFU of each kind.
- `et-trace gaps T [--largest N]`: idle time of the compute stream, inside steps and between them, the largest
  gaps with the steps around them and the collective that ended last, and how far the host runs ahead.
- `et-trace timeline T MS [MS ...] [--span MS]`: every kernel on every stream around given times;
  `et-trace timeline T --png gantt.png [--from MS --to MS]`: the Gantt chart.
- `et-deps graph.json T --from MS --to MS [--png deps.png]`: the steps of a range with their dependencies, and what
  each one waited for (a dependency, or the host).
- `et-trace idle T`: compute-stream idle time by the step that ended last elsewhere and by the step that resumed.
- `et-trace diff A B`: GPU time per kind in two traces, largest change first.
- `et-trace ranks-extract KIND T...` then `et-trace ranks-compare calls.jsonl`: which rank arrives last at an
  exchange and whether its host launched late.
- `et-memsnap memory_rank0.pickle`: what holds device memory at the peak of a recorded step.

Times are ms from the first kernel of the compute stream. Large traces need several GB of memory: analyze them on a
machine that has it.
