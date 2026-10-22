"""The loop: the scripted, passive side of an agent's run toward a goal (loop.py), served on a UNIX
socket (server.py), built from implementations of the abstract classes in abstract/.

Layout
    abstract/        GpuProvider, Checker
    gpu_provider/    implementations of GpuProvider (LocalGpus)
    checker/         implementations of Checker (CommandChecker)
    loop.py          Loop: the protocol, the state and the log of a run
    server.py        serve() and request()

Public components
    Loop, serve, request
"""
from explicit_trainer.loop.loop import Loop
from explicit_trainer.loop.server import request, serve

__all__ = ["Loop", "request", "serve"]
