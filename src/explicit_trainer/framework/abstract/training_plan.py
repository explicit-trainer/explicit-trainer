"""The training plan: what one training step does, as one static graph of steps.

Purpose and design. A plan joins the graphs of every call of every implemented module (variables,
forward and backward, for every micro-batch) with its own steps: making the copies the placement
declares (e.g. gathering sharded weights), moving values between the layouts of their producer and
their consumer, summing gradients of values with several consumers, reducing gradients and
updating state. It says what depends on what, nothing about when: the order is a schedule's.
The graph is the same in every training step; data-dependent sizes are host values that later
steps take as arguments.

The base class walks the model once. It binds every module's variables (env, evaluated top-down
from the root's bind; a variable bound only while running, by a let, is None) and records the
calls the root's forward makes, in order (self.trace). A call of a leaf, or of a composite the plan
implements as a whole, is one Call; a composite the plan substitutes runs its substitute instead
(an equivalent composite of coarse leaves, module.py), whose children are then traced.

Public components
    Handle(tensor)
        A value flowing between modules while tracing: a root input, or an output of a call;
        tensor is the global id of the port that made it.
    Call(module, inputs, outputs)
        One call: the module and its input and output Handles by port name.
    Let(name, value)
        An internal variable bound to a Handle.
    Tracer(root, atomic, substitute), .ops, .inputs, .outputs
        Runs the root's forward on Handles. atomic(module): whether a call is one Call;
        substitute(module): an equivalent composite to run in its place, or None. ops: the Calls
        and Lets in order; inputs / outputs: the root's Handles. call() and let() are its
        interpreter methods.
    TrainingPlan(model, placement, coord, micro_batches)
        coord: this device's index along every mesh dimension. Subclasses implement build() and
        may override implements() and substitute(). After build: outputs (name -> Value: what a
        training step returns, e.g. the loss) and calls ([module, inputs, outputs, output grads,
        input grads], the Values of every call by port name, used by checks).
    implements(module) -> bool
        Whether the plan implements this composite as a whole (default: no composite).
    substitute(module) -> CompositeModule | None
        An equivalent composite to run in place of the module, with its id and ports, whose leaves
        own no new state (default: none).
    build() -> Graph
        The graph of one training step on this device. Its Input steps name the root's inputs per
        micro-batch, and inputs of the whole step (micro-batch None).

Example
    class SimplePlan(TrainingPlan):
        def build(self):
            with Graph() as g:
                for i in range(self.micro_batches):
                    values = {h.tensor: input(name, i) for name, h in self.trace.inputs.items()}
                    ...                             # each Call of self.trace.ops through its impl
                self.outputs = {"loss": loss}
            return g

    plan = SimplePlan(model, placement, coord={"gpu": 0}, micro_batches=8)
    graph = plan.build()
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import sympy

from explicit_trainer.framework.abstract.module import CompositeModule, LeafModule, Module, Reference
from explicit_trainer.framework.abstract.placement import Placement
from explicit_trainer.framework.abstract.step import Graph


@dataclass(eq=False)
class Handle:
    """A value flowing between modules: an input of the root, or an output of a leaf call."""

    tensor: str  # global tensor id of the port that made it


@dataclass
class Call:
    module: Module  # a leaf, or a composite implemented as a whole
    inputs: dict[str, Handle]
    outputs: dict[str, Handle]


@dataclass
class Let:
    name: str  # internal variable of a composite
    value: Handle


class Tracer:
    def __init__(self, root: CompositeModule, atomic, substitute):
        """atomic(module): whether a call of the module is one call (a leaf, or a composite the plan
        implements as a whole) rather than the calls of its children; substitute(module): an equivalent
        composite to run in its place, or None."""
        self.atomic, self.substitute = atomic, substitute
        self.ops: list[Call | Let] = []
        self.inputs = {k: Handle(root.tensor_id(k)) for k in root.inputs}
        if root.substitute is None:
            sub = substitute(root)
            if sub is not None:
                object.__setattr__(root, "substitute", sub)
                _walk(sub, root.bind)
        previous, Module.interpreter = Module.interpreter, self
        try:
            out = (root.substitute or root).forward(**self.inputs)
        finally:
            Module.interpreter = previous
        self.outputs = out if isinstance(out, dict) else {next(iter(root.outputs)): out}

    def call(self, module: Module, *args, **kwargs):
        if module.substitute is None and not isinstance(module, LeafModule):
            sub = self.substitute(module)
            if sub is not None:
                object.__setattr__(sub, "parent", module.parent)  # same global id, not a child
                object.__setattr__(module, "substitute", sub)
                _walk(sub, module.parent.env)
        if module.substitute is not None:
            return module.substitute.forward(*args, **kwargs)
        if not self.atomic(module):
            return module.forward(*args, **kwargs)
        given = dict(zip(module.inputs, args)) | {k: v for k, v in kwargs.items() if v is not None}
        outs = {k: Handle(module.tensor_id(k)) for k in module.outputs}
        self.ops.append(Call(module, given, outs))
        return next(iter(outs.values())) if len(outs) == 1 else dict(outs)

    def let(self, composite: CompositeModule, **values):
        self.ops += [Let(k, v) for k, v in values.items()]


def _walk(m: Module, outer: dict):
    """Sets `env` on m and every module below it: its variables evaluated from the outer module's
    env; a variable bound only while running (by a let) is None."""
    subs = {sympy.Symbol(k, integer=True, nonnegative=True): v for k, v in outer.items() if isinstance(v, int)}
    m.env = {}
    for k in m.vars:
        e = m.bind[k]
        if isinstance(e, sympy.Symbol) and not isinstance(outer.get(e.name), int):
            m.env[k] = None
        else:
            m.env[k] = int(sympy.sympify(e).subs(subs))
    for child in m.children.values():
        _walk(child, m.env)


def _bind(root: Module):
    """Sets `env` on every module, evaluated top-down from the root's `bind`."""
    _walk(root, root.bind)


class TrainingPlan(ABC):
    def __init__(self, model: CompositeModule, placement: Placement, coord: dict[str, int], micro_batches: int):
        """coord: this device's index along every mesh dimension of the placement."""
        self.model, self.placement, self.coord, self.micro_batches = model, placement, coord, micro_batches
        _bind(model)
        self.trace = Tracer(model, lambda m: isinstance(m, LeafModule) or self.implements(m), self.substitute)

    def implements(self, module: CompositeModule) -> bool:
        """Whether the plan implements this composite as a whole (default: none)."""
        return False

    def substitute(self, module: CompositeModule) -> CompositeModule | None:
        """An equivalent composite to run in place of the module (default: none). It has the module's
        id and ports; its leaves own no new state (CoarseLeaf), so placement and state are unchanged."""
        return None

    outputs: dict  # name -> Value: what a step returns (e.g. the loss), set by build
    calls: list  # [module, inputs, outputs, output grads, input grads]: the Values of every call, by port
    # name (the grads those its backward received and returned), set by build

    @abstractmethod
    def build(self) -> Graph:
        """The graph of one training step on this device. Its Input steps name the inputs of the
        model's root module per micro-batch, and inputs of the whole step (micro-batch None)."""
