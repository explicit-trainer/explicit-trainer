"""Orders of a schedule and the search.

Public components
    OrderTest

Example
    python -m unittest tests.test_order
"""
import unittest

from explicit_trainer.framework.schedule import check, describe
from explicit_trainer.tools.order import critical_path, merged, schedule_of, search
from examples.tiny.schedule import generate
from tests.test_schedule import tiny_graph


class OrderTest(unittest.TestCase):
    def setUp(self):
        graph = tiny_graph(micro_batches=4)
        self.info, self.schedule = describe(graph), generate(graph)
        n = len(self.info["steps"])
        self.host, self.device = [1e-5] * n, [1e-3 * (1 + i % 3) for i in range(n)]
        self.streams = {i: q for q, m in self.schedule["queues"].items() for i in m}

    def test_orders_are_valid(self):
        order = merged(self.info, self.schedule["queues"])
        self.assertEqual(sorted(order), list(range(len(self.info["steps"]))))
        check(schedule_of(self.info, self.streams, order), self.info)
        cp = critical_path(self.info, order, self.host, self.device, group=2)
        check(schedule_of(self.info, self.streams, cp), self.info)

    def test_search(self):
        best, scores = search(self.info, self.schedule, self.host, self.device, group=1, minutes=0.02,
                              log=lambda s: None)
        check(best, self.info)
        self.assertLessEqual(scores["best"], scores["given"])


if __name__ == "__main__":
    unittest.main()
