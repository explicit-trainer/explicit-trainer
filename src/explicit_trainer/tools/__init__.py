"""Offline tools: they read graphs (framework.schedule.describe), schedules, profiled steps and memory
snapshots, and need no GPU.

Modules
    steptrace.py    reading a profiled step with step annotations
    simulate.py     a training step under a schedule, simulated with profiled step times
    order.py        searching a schedule's order on simulated steps
    deps.py         steps of a time range with their dependencies
    steps.py        GPU time and MFU per kind of step
    gaps.py         idle time of the compute stream
    timeline.py     every stream around given times, or as a Gantt chart
    trace_idle.py   idle time by cause
    trace_diff.py   GPU time per kind in two traces
    trace_ranks.py  which rank arrives last at an exchange
    memsnap.py      device memory at the peak of a recorded step
    stub.py         running code with GPU-only packages stubbed
"""
