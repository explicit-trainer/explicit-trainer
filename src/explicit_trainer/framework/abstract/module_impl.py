"""Implementations of modules under a placement, as graphs of steps.

Purpose and design. A ModuleImpl implements one module, a leaf or a composite, for the copies its
placement gives the module's ports and the states of its subtree. Its forward and backward build
graphs of steps (step.py) rather than computing: once every variable has a value, what a module
does is fixed, so it is a static graph that the framework runs later, interleaved with others as
the dependencies allow. A good implementation has one compute step on each side of each
communication; inside a step it may do anything that needs only local data.

Values enter its graphs as the module's inputs (Values of the "value" copies of its ports) and as
copies of states (read), and leave as outputs, input gradients, or writes into state copies. A copy
in an FP8 dtype is a pair (values, scales). Sizes that depend on data and that the implementation
needs beyond the model's variables (e.g. how many rows go to each device of an exchange) come from
its own graph, variables(): the plan brings each of them to the host (to_host) and binds the host
Value in self.env before building forward, whose steps take it as an argument.

Public components
    ModuleImpl(module, copies, env, mesh, coord), .module_type, .micro_batch, .group
        copies: tensor name relative to the module ("w", or "gate.w" inside a coarse leaf) -> its
        copies; env: the variables' values; mesh: size of every mesh dimension; coord: this
        device's index along each. micro_batch, group (the first micro-batch of the group of
        micro-batches run together): set by the plan while it builds a call's graphs.
    accepts(module, copies) -> bool
        Whether this class implements the module (by default: isinstance of module_type).
    read(state, copy="value") -> CopyRef
    write(state, copy, x, accumulate=True)
        Access a copy of a state of the module or below it, by relative name; accesses to copies
        that live shorter than the training step carry the current micro-batch (the current
        group's first micro-batch for copies of lifetime "group").
    variables(**inputs) -> (variables, values)
        Both by name: the data-dependent sizes the forward needs, and Values to pass to forward
        besides the inputs. Default: none.
    forward(**inputs) -> (outputs, saved)
        Outputs by port name, and what the backward needs (any Values, nested in dicts).
    backward(saved, **grads) -> input gradients
        Receives the gradients of the floating outputs; returns those of the floating inputs by
        name, and writes state gradients into their "grad" copies.

Example
    @step
    def scale_fwd(x, w):
        return (x * w.float()).to(x.dtype)

    @step(outputs=2)
    def scale_bwd(x, w, dy):
        return (dy * w.float()).to(dy.dtype), (dy.float() * x.float()).flatten(0, -2).sum(0)

    class ScaleImpl(ModuleImpl):
        module_type = Scale                       # the leaf of module.py's example

        def forward(self, x):
            w = self.read("w", "compute")
            return {"y": scale_fwd(x, w)}, {"x": x, "w": w}

        def backward(self, saved, y):
            dx, dw = scale_bwd(saved["x"], saved["w"], y)
            self.write("w", "grad", dw)
            return {"x": dx}
"""
from explicit_trainer.framework.abstract.module import Module
from explicit_trainer.framework.abstract.placement import Copy
from explicit_trainer.framework.abstract.step import Value, read, write


class ModuleImpl:
    module_type: type = Module

    def __init__(self, module: Module, copies: dict[str, dict[str, Copy]], env: dict, mesh: dict[str, int],
                 coord: dict[str, int]):
        """copies: relative tensor id -> its copies (the placement's), for the module's ports and the
        states of its subtree; env: the variables' values; mesh: size of every mesh dimension; coord:
        this device's index along each."""
        self.module, self.copies, self.env, self.mesh, self.coord = module, copies, env, mesh, coord
        self.micro_batch = self.group = None  # set by the plan while it builds this call's graphs

    @classmethod
    def accepts(cls, module: Module, copies: dict[str, dict[str, Copy]]) -> bool:
        return isinstance(module, cls.module_type)

    def read(self, state: str, copy: str = "value") -> Value:
        """A copy of a state of the module or below it, by relative tensor id."""
        return read(self._tensor(state), copy, self._micro_batch_of(state, copy))

    def write(self, state: str, copy: str, x, accumulate: bool = True):
        """Writes (adds, when `accumulate`) x into a copy of a state of the module or below it."""
        write(self._tensor(state), copy, x, accumulate, self._micro_batch_of(state, copy))

    def _tensor(self, state: str) -> str:
        """The global id of a state named relative to the module (a coarse leaf's are the originals')."""
        m = self.module
        return m.tensor_id(state) if state in getattr(m, "state", {}) else f"{m.global_id}.{state}"

    def _micro_batch_of(self, state: str, copy: str) -> int | None:
        """The micro-batch an access carries: only copies that live shorter than the step have one
        (a group's copies: its first micro-batch)."""
        lifetime = self.copies[state][copy].lifetime
        if lifetime == "group":
            return self.group
        return self.micro_batch if lifetime in ("micro_batch", "call", "backward") else None

    def variables(self, **inputs: Value) -> tuple[dict[str, Value], dict[str, Value]]:
        return {}, {}

    def forward(self, **inputs: Value) -> tuple[dict[str, Value], dict[str, Value]]:
        raise NotImplementedError

    def backward(self, saved: dict[str, Value], **grads: Value) -> dict[str, Value]:
        raise NotImplementedError
