"""Steps: the vocabulary of every graph the framework runs.

Purpose and design. All work of a training step is a step: a local computation or a collective.
An implementation of a module is a graph of steps, and so is the plan of a whole training step.
A graph is written as ordinary code over Values (handles, not data); running that code inside a
Graph only records the steps, and the framework runs them later, in the order of a schedule.
The edges of a graph are the Values a step reads and the order of accesses to each copy; nothing
else. A graph is static: a data-dependent size is a host value (to_host, or a step on host values)
that later steps take as an argument, so only sizes, not the shape of the graph, depend on data.

Copies of tensors (placement.py) are not steps; steps access them. read() gives a CopyRef that a
step takes as an argument and reads as the copy is when the step starts; write() attaches a write
to the step that makes the written Values, applied when that step finishes. Accesses to one copy
are ordered by program order: a step reading a copy follows the steps writing it before; a step
writing it follows those reading or writing it before. A copy that lives shorter than a training
step (micro_batch, call, backward) exists once per micro-batch, so its accesses carry the
micro-batch and only accesses of the same micro-batch are ordered.

Public components
    Value(step, index)
        Output `index` of `step`; the handle code passes around. Never holds data.
    CopyRef(tensor, copy, micro_batch=None), .key
        A copy of a tensor (global tensor id, copy name), as a step argument; key is the access key
        (tensor, copy, micro_batch) by which accesses are ordered and copies are stored.
    Write(copy, x, accumulate=True)
        A write of x (a Value, or a tuple of Values of one step) into a CopyRef when its step ends.
    Step(args, outputs=1, writes=[]), .values(), .reads()
        Base of all steps. args: Values, CopyRefs, tuples of them (an FP8 pair), and constants.
        values() and reads() list the Values and CopyRefs among the args (at any tuple depth).
    ComputeStep(fn=...)
        fn(*args) on one device; on the host when every Value it reads is a host value. It reads
        only its arguments and frees all its temporary memory when it returns.
    CommunicationStep(dims=...) and its kinds
        AllGather(axis), ReduceScatter(axis), AllReduce(op = "sum" | "max"), AllToAll (args x,
        send, recv: rows sent to and received from each member, host lists or host Values). A
        collective runs over the group of devices spanning mesh dimensions `dims`, along the first
        dimension of its tensor (the axis it names is the outermost in memory).
    ToHost
        The value of a small tensor on the host (a list), e.g. split sizes.
    Input(name, micro_batch)
        A tensor given from outside: input `name` of micro-batch `micro_batch` (None: of the whole
        training step, e.g. the step number).
    Graph(), .steps, .add(step), Graph.current
        Records the steps made while it is the current graph (`with Graph() as g:`); add() appends
        a step and returns its Value (or a tuple of Values).
    step(fn, *, outputs=1)
        Marks fn as a compute step: calling it records a ComputeStep and returns its Values instead
        of running it. outputs is a count (fn returns a tuple, or one value) or a list of names (fn
        returns a dict; the call returns a dict of Values). Arguments go by position or by name;
        a dict of arguments reaches fn as a dict. step(f, outputs=[...])(...) makes a step of a
        function whose outputs are known only at the call.
    all_gather(x, dims, axis), reduce_scatter(x, dims, axis), all_reduce(x, dims, op="sum"),
    all_to_all(x, dims, send, recv), to_host(x), input(name, micro_batch)
        Record the step of that kind and return its Value.
    read(tensor, copy, micro_batch=None) -> CopyRef
    write(tensor, copy, x, accumulate=True, micro_batch=None)
        Access a copy; write() requires all of x to come from one step.

Example
    @step
    def scaled(x, factor):            # runs later, on the device
        return x * factor

    @step(outputs=["y", "h"])
    def block(x, w):
        h = x @ w.t()
        return {"y": h.relu(), "h": h}

    with Graph() as g:
        x = input("tokens", 0)
        w = read("model.w", "compute")
        out = block(x, w=w)                       # {"y": Value, "h": Value}
        y = all_reduce(out["y"], ("node", "gpu"))
        write("model.w", "grad", scaled(y, 2.0))  # attached to the step scaled() made
"""
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(eq=False)
class Value:
    """A tensor in a graph: made by `step`, its output number `index`."""

    step: "Step"
    index: int = 0


@dataclass(frozen=True)
class CopyRef:
    """A reference to a copy of a tensor (by global tensor id and copy name), an argument of a step."""

    tensor: str
    copy: str
    micro_batch: int | None = None

    @property
    def key(self) -> tuple:
        return self.tensor, self.copy, self.micro_batch


@dataclass(eq=False)
class Write:
    """A write of a step's output(s) `x` into a copy, when the step finishes."""

    copy: CopyRef
    x: object  # a Value or a tuple of Values of the step
    accumulate: bool = True


def _flat(args) -> list:
    """The arguments of a step, tuples (of any depth) flattened."""
    return [y for a in args for y in (_flat(a) if isinstance(a, tuple) else (a,))]


@dataclass(eq=False)
class Step:
    """One step: what it runs on (its args: Values, the edges of the graph, tuples of Values,
    and constants) and how many Values it makes."""

    args: tuple
    outputs: int = 1
    writes: list = field(default_factory=list)  # Write: copies written when the step finishes

    def values(self) -> list[Value]:
        """The Values the step reads: the edges into it."""
        return [v for v in _flat(self.args) if isinstance(v, Value)]

    def reads(self) -> list[CopyRef]:
        """The copies the step reads."""
        return [c for c in _flat(self.args) if isinstance(c, CopyRef)]


@dataclass(eq=False)
class ComputeStep(Step):
    """fn(*args) on one device."""

    fn: Callable = None


@dataclass(eq=False)
class CommunicationStep(Step):
    """A collective over the group of devices spanning mesh dimensions `dims`."""

    dims: tuple[str, ...] = ()


@dataclass(eq=False)
class AllGather(CommunicationStep):
    axis: str = ""  # the blocks of this axis, concatenated in mesh order on every device


@dataclass(eq=False)
class ReduceScatter(CommunicationStep):
    axis: str = ""  # the sum over the group, each device keeping its block of this axis


@dataclass(eq=False)
class AllReduce(CommunicationStep):
    op: str = "sum"  # sum or max over the group, on every device


@dataclass(eq=False)
class AllToAll(CommunicationStep):
    """args = (x, send, recv): rows of x, send[j] to member j of the group, recv[j] from it; the
    splits are host lists, constants or Values (made on the host). Rows of x after the first
    sum(send) are not sent, so x may be padded to a size known without the device."""


@dataclass(eq=False)
class ToHost(Step):
    """The value of a small tensor on the host (a list), to bind a variable."""


@dataclass(eq=False)
class Input(Step):
    """A tensor given to the step from outside: input `name` of micro-batch `micro_batch`, or of the
    whole step (None)."""

    name: str = ""
    micro_batch: int | None = None


class Graph:
    """The steps one piece of code made, in the order it made them."""

    current: "Graph | None" = None

    def __init__(self):
        self.steps: list[Step] = []

    def add(self, step: Step):
        self.steps.append(step)
        out = tuple(Value(step, i) for i in range(step.outputs))
        return out[0] if step.outputs == 1 else out

    def __enter__(self):
        self.outer, Graph.current = Graph.current, self
        return self

    def __exit__(self, *exc):
        Graph.current = self.outer


def step(fn: Callable = None, *, outputs: int | list[str] = 1):
    """Marks fn as a step: calling it records a ComputeStep in the graph being built (fn runs later,
    on this device, or on the host when its Value arguments are host values) and returns its Values.
    outputs: how many (fn returns a tuple of that many; one Value, or a tuple, is returned), or their
    names (fn returns a dict with those keys; a dict of Values is returned). Arguments may be passed
    by position or by name; dicts of arguments reach fn as dicts. Used as @step, @step(outputs=...),
    or step(f, outputs=...)(...) for a function whose outputs are known only when it is called."""
    def wrap(f):
        names = list(outputs) if isinstance(outputs, list) else None
        n = len(names) if names is not None else outputs

        def call(*args, **kwargs):
            keys = list(kwargs)
            dicts = {k: list(v) for k, v in kwargs.items() if isinstance(v, dict)}
            flat = args + tuple(tuple(v.values()) if k in dicts else v for k, v in kwargs.items())

            def run(*xs):
                given = {k: dict(zip(dicts[k], x)) if k in dicts else x for k, x in zip(keys, xs[len(args):])}
                out = f(*xs[: len(args)], **given)
                if names is None:
                    return out
                return tuple(out[k] for k in names) if n > 1 else out[names[0]]
            run.__qualname__, run.__module__ = f.__qualname__, f.__module__
            made = Graph.current.add(ComputeStep(flat, n, fn=run))
            if names is None:
                return made
            return dict(zip(names, made if n > 1 else (made,)))
        call.__wrapped__ = f
        return call
    return wrap(fn) if fn is not None else wrap


def all_gather(x: Value, dims: tuple[str, ...], axis: str) -> Value:
    return Graph.current.add(AllGather((x,), dims=dims, axis=axis))


def reduce_scatter(x: Value, dims: tuple[str, ...], axis: str) -> Value:
    return Graph.current.add(ReduceScatter((x,), dims=dims, axis=axis))


def all_reduce(x: Value, dims: tuple[str, ...], op: str = "sum") -> Value:
    return Graph.current.add(AllReduce((x,), dims=dims, op=op))


def all_to_all(x: Value, dims: tuple[str, ...], send, recv) -> Value:
    return Graph.current.add(AllToAll((x, send, recv), dims=dims))


def to_host(x: Value) -> Value:
    return Graph.current.add(ToHost((x,)))


def input(name: str, micro_batch: int | None) -> Value:
    return Graph.current.add(Input((), name=name, micro_batch=micro_batch))


def read(tensor: str, copy: str, micro_batch: int | None = None) -> CopyRef:
    return CopyRef(tensor, copy, micro_batch)


def write(tensor: str, copy: str, x, accumulate: bool = True, micro_batch: int | None = None):
    steps = {v.step for v in (x if isinstance(x, tuple) else (x,))}
    assert len(steps) == 1, ("a write takes outputs of one step", tensor, copy)
    steps.pop().writes.append(Write(CopyRef(tensor, copy, micro_batch), x, accumulate))
