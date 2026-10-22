"""The command-line entry points of explicit-trainer (pyproject.toml, [project.scripts]): thin wrappers
that parse arguments and call the library. Without installing the package, each also runs as
`python -m explicit_trainer.cli.<module>` with src/ on PYTHONPATH.

Public components
    et-schedule     schedule.py: validate schedules, estimate a step under them
    et-order        order.py: search a schedule's order on simulated steps
    et-deps         deps.py: steps of a time range with their dependencies
    et-trace        trace.py: reports on profiled steps
    et-memsnap      memsnap.py: device memory at the peak of a recorded step
    et-stub         stub.py: run a script with GPU-only packages stubbed
    et-skills       skills.py: list or install the skills
    et-loop         loop.py: talk to a served loop
"""
