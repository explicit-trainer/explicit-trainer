"""Skills: instructions for agents that work with explicit-trainer, one Markdown file per skill in this
directory (Claude Code skill format: a front matter with name and description, then the text).

Purpose and design. The skills are flat files here so that they ship with the package; install()
copies them where an agent's tooling finds them: <dest>/<name>/SKILL.md (e.g. dest = a project's
.claude/skills).

Public components
    names() -> [str]
        The skills, by name (the file name without .md), sorted.
    text(name) -> str
        A skill's file.
    install(dest) -> [path]
        Writes every skill as <dest>/<name>/SKILL.md and returns the paths written.

Example
    install(".claude/skills")
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def names() -> list[str]:
    return sorted(f[:-3] for f in os.listdir(_HERE) if f.endswith(".md"))


def text(name: str) -> str:
    return open(os.path.join(_HERE, name + ".md")).read()


def install(dest: str) -> list[str]:
    out = []
    for name in names():
        os.makedirs(os.path.join(dest, name), exist_ok=True)
        path = os.path.join(dest, name, "SKILL.md")
        with open(path, "w") as f:
            f.write(text(name))
        out.append(path)
    return out
