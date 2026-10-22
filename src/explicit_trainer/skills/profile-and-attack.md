---
name: profile-and-attack
description: The optimization loop for a training step in explicit-trainer - measure, find the largest loss of time or memory, change one thing, check correctness, measure again. Use when asked to make a training step faster or fit in memory.
---

# Profile and attack

1. **Measure.** Run the step on the GPUs (check that they are free first), with one profiled step and the step time
   and memory of the timed steps. The first step of a run executes serially (`step(serial=True)`) and is not timed.
2. **Find the target** (skill `read-trace`): the largest of
   - a kind of compute step with low MFU or many kernels per call (`et-trace steps`);
   - idle time of the compute stream between steps (`et-trace gaps`, `et-deps`): waiting for a collective, for the
     host, or for a dependency the schedule placed late;
   - a slow rank (`et-trace ranks-*`);
   - memory at the peak (`et-memsnap`), when memory limits the configuration.
3. **Change one thing**, the one that removes the most time for the least code:
   - idle time from order: a schedule change (skills `write-schedule`, `validate-and-simulate`), no code change;
   - many small kernels in a step: fuse inside the step (one compute step on each side of each communication);
   - a slow kernel: a better kernel for that step;
   - communication on the critical path: overlap it, move it to another stream, or change the placement.
4. **Check correctness** with `Trainer.check` next to the references after any change of the step; keep the error
   levels the task documents. Never change precision unless the task says so.
5. **Measure again** with the same configuration; keep the change only if the step time (or memory) improved beyond
   the run-to-run noise. Record what was tried and the numbers, including what did not work.

When normal changes stop helping, try larger ones: different placements, coarse leaves that make a large part of the
model one step, custom kernels for the heaviest steps.
