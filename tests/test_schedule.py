"""describe, check and simulate on the example's graph.

Public components
    tiny_graph(micro_batches=2) -> Graph
        The example's graph on one device (also used by the other tests).
    ScheduleTest

Example
    python -m unittest tests.test_schedule
"""
import unittest

from explicit_trainer.framework.schedule import check, describe
from explicit_trainer.tools.simulate import critical, report, simulate
from examples.tiny.model import Net
from examples.tiny.placement import DataParallel
from examples.tiny.plan import TinyPlan
from examples.tiny.schedule import generate


def tiny_graph(micro_batches: int = 2):
    net = Net(16, 32, 8, id="net", batch=32)
    plan = TinyPlan(net, DataParallel(net, 1), {"gpu": 0}, micro_batches)
    return plan.build()


class ScheduleTest(unittest.TestCase):
    def setUp(self):
        self.graph = tiny_graph()
        self.info = describe(self.graph)
        self.schedule = generate(self.graph)

    def test_describe(self):
        types = {s["type"] for s in self.info["steps"]}
        self.assertEqual(types, {"input", "compute", "collective"})
        self.assertEqual(self.info["hash"], describe(tiny_graph())["hash"])
        for i, s in enumerate(self.info["steps"]):
            self.assertTrue(all(d < i for d in s["deps"]))

    def test_check_accepts_program_order(self):
        check(self.schedule, self.info)

    def test_check_refusals(self):
        other = {"graph": "0" * 40, "queues": self.schedule["queues"]}
        self.assertRaises(AssertionError, check, other, self.info)
        queues = {q: list(m) for q, m in self.schedule["queues"].items()}
        queues["compute"] = queues["compute"][1:]
        self.assertRaises(AssertionError, check, {"graph": self.info["hash"], "queues": queues}, self.info)
        comm = self.schedule["queues"]["comm"]
        moved = {"compute": self.schedule["queues"]["compute"] + comm, "comm": []}
        self.assertRaises(AssertionError, check, {"graph": self.info["hash"], "queues": moved}, self.info)
        stuck = {q: list(reversed(m)) for q, m in self.schedule["queues"].items()}
        self.assertRaises(AssertionError, check, {"graph": self.info["hash"], "queues": stuck}, self.info)

    def test_simulate(self):
        n = len(self.info["steps"])
        sim = simulate(self.info, self.schedule, [1e-5] * n, [1e-3] * n)
        compute = sum(1 for s in self.schedule["queues"]["compute"] if self.info["steps"][s]["type"] != "input")
        self.assertGreaterEqual(sim["step_s"], compute * 1e-3)
        path = critical(self.info, sim)
        self.assertEqual(sim["end"][path[0]], max(sim["end"]))
        self.assertIn("simulated step", report(self.info, sim))


if __name__ == "__main__":
    unittest.main()
