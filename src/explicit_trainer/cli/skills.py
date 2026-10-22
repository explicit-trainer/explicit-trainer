"""et-skills: list the skills that ship with explicit-trainer, or install them for an agent.

et-skills list
et-skills install <dest>          e.g. .claude/skills: writes <dest>/<name>/SKILL.md

Public components
    main()

Example
    et-skills install .claude/skills
"""
import argparse

from explicit_trainer import skills


def main():
    p = argparse.ArgumentParser(prog="et-skills")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    s = sub.add_parser("install")
    s.add_argument("dest")
    a = p.parse_args()
    if a.command == "list":
        print("\n".join(skills.names()))
    else:
        print("\n".join(skills.install(a.dest)))


if __name__ == "__main__":
    main()
