"""et-memsnap: what holds device memory at the peak of a recorded step.

et-memsnap <memory_rank<N>.pickle> [--rows N]

Prints the report of tools/memsnap.py.

Public components
    main()

Example
    et-memsnap memory_rank0.pickle --rows 25
"""
import argparse

from explicit_trainer.tools.memsnap import report


def main():
    p = argparse.ArgumentParser(prog="et-memsnap")
    p.add_argument("snapshot")
    p.add_argument("--rows", type=int, default=20)
    a = p.parse_args()
    print(report(a.snapshot, a.rows))


if __name__ == "__main__":
    main()
