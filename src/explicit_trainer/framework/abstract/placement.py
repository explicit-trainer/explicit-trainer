"""Placement: every copy of every tensor of a model, and how each is laid out over a device mesh.

Purpose and design. A model (module.py) says what is computed; a placement says where every tensor
lives while it is computed. It gives each tensor one or more copies: the forms in which the
tensor's values (or its gradient's, or optimizer state) exist, each with its layout, lifetime and
the copy it is made from. Implementations read and write copies by name and hold no other form of
a tensor; what they allocate besides is workspace of a single step. So all memory and all data
movement of a training step follow from the placement and are known before the step runs.

Layout. The mesh is a set of named dimensions, e.g. {"node": 2, "gpu": 4}. For each axis of the
tensor a layout lists the mesh dimensions it is split over (outer first); over the other mesh
dimensions, unless `partial`, the copy is replicated. Along a `partial` dimension each device
holds one summand, and the value is their sum.
    A split of n indices over k devices gives device i the block [i * ceil(n/k), (i + 1) *
    ceil(n/k)), clipped to n: the last blocks are shorter or empty. A ragged axis split together
    with the data it came from keeps each row on the device that produced it; split by the axis
    before it, each device holds whole groups, in an order the copy's source defines.
order: the axes in memory order, outermost first (None: declaration order). align: an axis padded
to a multiple of the given size. dtype: the dtype it is held in (None: the declared one); an FP8
copy carries FP32 scales, one per index of the axes not in `scale_over`, shared over those in it.
memory: device or host.

Copy. lifetime: persistent (across training steps), step, group (one pass of a group of
micro-batches run together: their forward, or their backward; a copy rewritten whole for the next
pass is a new one), micro_batch, call (one call of the module), or backward (from a forward call to
the backward of the same micro-batch). source: the copy it is made from (None: written by a module,
or a persistent copy's initial value). select: the input whose values name the rows the copy holds.
Every tensor has a copy "value"; a trainable state has "grad" and the optimizer's copies; a port's
gradient is "grad" in the dual layout of its value (replicated becomes partial, split stays split,
partial becomes replicated).

Public components
    Layout(split={}, partial=(), order=None, align={}, dtype=None, scale_over=(), memory="device")
    Copy(name, layout, lifetime, source=None, select=None)
    Placement(model), .mesh, .place(tensor_id) -> dict[str, Copy]
        The copies of the tensor with this global id (a port or a state), by name. A placement is
        built for one model; subclasses implement place, usually from patterns of tensor ids.

Example
    class AllSharded(Placement):
        mesh = {"gpu": 4}

        def place(self, tensor_id):
            module, name = self.model.resolve(tensor_id)
            if name in getattr(module, "state", {}):
                first = next(iter(module.state[name].axes))
                rest = Layout({first: ("gpu",)}, dtype=torch.float32)
                return {"value": Copy("value", rest, "persistent"),
                        "compute": Copy("compute", Layout(dtype=torch.bfloat16), "step", "value"),
                        "grad": Copy("grad", Layout(partial=("gpu",)), "step")}
            return {"value": Copy("value", Layout({"batch": ("gpu",)}), "backward")}
"""
from dataclasses import dataclass, field

import torch

from explicit_trainer.framework.abstract.module import Module


@dataclass(frozen=True)
class Layout:
    split: dict[str, tuple[str, ...]] = field(default_factory=dict)  # axis -> mesh dims, outer first
    partial: tuple[str, ...] = ()
    order: tuple[str, ...] | None = None
    align: dict[str, int] = field(default_factory=dict)
    dtype: torch.dtype | None = None
    scale_over: tuple[str, ...] = ()  # FP8: axes that share one scale
    memory: str = "device"  # device or host


@dataclass(frozen=True)
class Copy:
    name: str
    layout: Layout
    lifetime: str  # persistent, step, group, micro_batch, call or backward
    source: str | None = None  # name of the copy this one is made from
    select: str | None = None  # the input of the module whose values name the rows held


class Placement:
    """The copies of every tensor of one model; built for that model (and its mesh)."""

    mesh: dict[str, int]

    def __init__(self, model: Module):
        self.model = model

    def place(self, tensor_id: str) -> dict[str, Copy]:
        """The copies of the tensor (an input, output or state of a module) with this global id,
        by name."""
        raise NotImplementedError
