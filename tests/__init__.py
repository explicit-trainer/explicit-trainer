"""Tests of explicit-trainer, with the standard library's unittest; they need no GPU.

python -m unittest discover -s tests -t .      (from the repository root, with the package installed)

Modules
    test_schedule.py    describe, check and simulate on the example's graph
    test_order.py       orders of a schedule and the search
    test_tiny.py        the example trains and matches its references (gloo, CPU)
    test_loop.py        the loop's protocol, state and log, and its socket server
    test_tools.py       stub, skills and the et-* commands
"""
