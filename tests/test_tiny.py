"""The example trains and matches its references, in one process with gloo on CPU.

Public components
    setUpModule()
        Initializes a gloo process group of one rank on a free local port.
    TinyTest

Example
    python -m unittest tests.test_tiny
"""
import os
import socket
import unittest

import torch.distributed as dist

from examples.tiny.run import main


def setUpModule():
    if not dist.is_initialized():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK="0", WORLD_SIZE="1")
        dist.init_process_group("gloo")


class TinyTest(unittest.TestCase):
    def test_train_and_check(self):
        losses, report = main(steps=5, micro_batches=2)
        self.assertLess(losses[-1], losses[0])
        plan_loss, ref_loss = report["loss"]
        self.assertAlmostEqual(plan_loss, ref_loss, delta=1e-4 * abs(ref_loss))
        for k in ("net.l1.w", "net.l2.w", "leaf Linear.y", "leaf Linear.dx", "leaf SquaredError.dy"):
            self.assertLess(report[k], 1e-5, k)

    def test_check_without_leaves(self):
        _, report = main(steps=1, micro_batches=1, leaves=False)
        self.assertEqual(set(report), {"loss", "net.l1.w", "net.l2.w"})
        self.assertLess(report["net.l2.w"], 1e-5)


if __name__ == "__main__":
    unittest.main()
