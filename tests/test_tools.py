"""stub, skills and the et-* commands.

Public components
    ToolsTest

Example
    python -m unittest tests.test_tools
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from explicit_trainer import skills
from explicit_trainer.cli import schedule as schedule_cli
from explicit_trainer.cli import skills as skills_cli
from explicit_trainer.framework.schedule import describe
from examples.tiny.schedule import generate
from tests.test_schedule import tiny_graph


def _run_cli(main, *argv) -> str:
    out = io.StringIO()
    with mock.patch.object(sys, "argv", ["et", *argv]), contextlib.redirect_stdout(out):
        main()
    return out.getvalue()


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_skills(self):
        names = skills.names()
        self.assertIn("write-plan", names)
        for path in skills.install(os.path.join(self.dir, "skills")):
            head = open(path).read().split("\n")
            self.assertEqual(head[0], "---")
            self.assertTrue(head[1].startswith("name: "))
            self.assertEqual(head[1][6:], os.path.basename(os.path.dirname(path)))
        self.assertEqual(_run_cli(skills_cli.main, "list").split(), names)

    def test_schedule_validate(self):
        graph = tiny_graph()
        g, s = os.path.join(self.dir, "graph.json"), os.path.join(self.dir, "schedule.json")
        json.dump(describe(graph), open(g, "w"))
        json.dump(generate(graph), open(s, "w"))
        self.assertIn("valid for this graph", _run_cli(schedule_cli.main, "validate", g, s))

    def test_stub(self):
        script = os.path.join(self.dir, "script.py")
        with open(script, "w") as f:
            f.write("import sys\nimport no_such_gpu_package.kernels as k\nprint(sys.argv[1:], k.anything)\n")
        out = subprocess.run([sys.executable, "-c", "from explicit_trainer.cli.stub import main; main()",
                              "--stub", "no_such_gpu_package", script, "--flag", "x"],
                             capture_output=True, text=True, check=True).stdout
        self.assertTrue(out.startswith("['--flag', 'x']"))


if __name__ == "__main__":
    unittest.main()
