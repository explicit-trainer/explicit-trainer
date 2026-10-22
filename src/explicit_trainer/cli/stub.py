"""et-stub: run a script on a machine without GPUs, with GPU-only packages replaced by stubs.

et-stub --stub <package>[,<package>...] <script> [args...]

The packages are top-level package names (e.g. triton,deep_ep); see tools/stub.py.

Public components
    main()

Example
    et-stub --stub triton,deep_ep build_schedule.py config.json
"""
import argparse

from explicit_trainer.tools.stub import run


def main():
    p = argparse.ArgumentParser(prog="et-stub")
    p.add_argument("--stub", required=True, help="comma-separated top-level packages to stub")
    p.add_argument("script")
    p.add_argument("args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    run(a.script, a.args, [x for x in a.stub.split(",") if x])


if __name__ == "__main__":
    main()
