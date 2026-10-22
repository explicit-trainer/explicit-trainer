"""Running code on a machine without GPUs, with GPU-only packages replaced by stubs: building a plan's
graph, generating and validating its schedule, simulating, analyzing.

Purpose and design. After install(), importing one of the given top-level packages gives an empty
module whose attributes are mocks, so a module that imports GPU kernels at its top can be imported;
nothing that would run them may be called. torch.compile returns the function unchanged. Which
packages are GPU-only depends on the model's code, so they are always named by the caller.

Public components
    install(packages)
        Replaces the top-level packages named in `packages` by stubs, for every later import.
    run(script, args, packages)
        install(packages), then runs the script as __main__ with sys.argv = [script, *args].

Example
    run("generate.py", ["config.json"], ["triton", "deep_ep"])
"""
import importlib.abc
import importlib.machinery
import runpy
import sys
from unittest import mock

import torch


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, packages: tuple[str, ...]):
        self.packages = packages

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self.packages:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = mock.MagicMock(name=spec.name)
        m.__path__, m.__spec__, m.__name__ = [], spec, spec.name
        return m

    def exec_module(self, module):
        pass


def _no_compile(fn=None, **_):
    return fn if fn is not None else (lambda f: f)


def install(packages: list[str]):
    loaded = [p for p in packages if p in sys.modules]
    assert not loaded, ("already imported, cannot be stubbed", loaded)
    sys.meta_path.insert(0, _Finder(tuple(packages)))
    torch.compile = _no_compile


def run(script: str, args: list[str], packages: list[str]):
    install(packages)
    sys.argv = [script, *args]
    runpy.run_path(script, run_name="__main__")
