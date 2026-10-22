"""Trains the example for a few steps and checks one step against the references.

torchrun --nproc-per-node 1 -m examples.tiny.run [--steps 5] [--micro-batches 2]     one process (gloo, CPU)
torchrun --nproc-per-node 2 -m examples.tiny.run                                     two processes

Purpose and design. Every rank makes the same data from a fixed seed and takes its part of the
batch; the trainer runs the plan's graph under the generated schedule, on the same batch every
step, so the loss falls. The check runs one more step, on another batch, next to the model's
references and prints the relative errors of the loss, the weight gradients and every call (expected
below 1e-5 in FP32).

Public components
    main(steps, micro_batches, leaves=True) -> (losses, check report)
        Initializes the process group (gloo, from the environment torchrun sets) unless it is.
        leaves: also compare every call of every implemented module (Trainer.check).

Example
    losses, report = main(steps=5, micro_batches=2)
"""
import argparse

import torch
import torch.distributed as dist

from explicit_trainer.framework.trainer import Mesh, Trainer
from examples.tiny.model import Net
from examples.tiny.placement import DataParallel
from examples.tiny.plan import TinyPlan
from examples.tiny.schedule import generate

BATCH, D_IN, D_HIDDEN, D_OUT = 32, 16, 32, 8


def _batches(micro_batches: int, rank: int, world: int, seed: int) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    rows = BATCH // world
    out = []
    for _ in range(micro_batches):
        x, target = torch.randn(BATCH, D_IN, generator=g), torch.randn(BATCH, D_OUT, generator=g)
        out.append({"x": x[rank * rows:(rank + 1) * rows], "target": target[rank * rows:(rank + 1) * rows]})
    return out


def main(steps: int, micro_batches: int, leaves: bool = True):
    if not dist.is_initialized():
        dist.init_process_group("gloo")
    mesh = Mesh({"gpu": dist.get_world_size()})
    net = Net(D_IN, D_HIDDEN, D_OUT, id="net", batch=BATCH)
    plan = TinyPlan(net, DataParallel(net, mesh.world), mesh.coord, micro_batches)
    trainer = Trainer(plan, generate(plan.build()), mesh)
    batches = _batches(micro_batches, mesh.rank, mesh.world, seed=0)
    losses = [trainer.step(batches, serial=s == 0)["loss"].item() for s in range(steps)]
    grads = {w: "grad" for w in plan.weights}
    batches = _batches(micro_batches, mesh.rank, mesh.world, seed=1)
    report = trainer.check(batches, grads, tokens="target", leaves=leaves)
    return losses, report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--micro-batches", type=int, default=2)
    a = p.parse_args()
    losses, report = main(a.steps, a.micro_batches)
    if dist.get_rank() == 0:
        print("losses", " ".join(f"{x:.4f}" for x in losses))
        for k, v in report.items():
            print(f"{k}: {v}")
    dist.destroy_process_group()
