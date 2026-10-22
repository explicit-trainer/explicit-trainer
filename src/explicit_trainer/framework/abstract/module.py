"""A language for describing models as trees of modules.

Purpose and design. A model is written once, as a specification: what each part computes, with
which tensors, in which shapes. How it is run (placement, implementations, schedule) is not part
of it. Every module has an explicit id and a place in a tree; every tensor is named by its global
id, the module's path and the tensor's name ("transformer.layer-3.attn.wq_a.w"), so placements and
implementations refer to tensors without ambiguity. A tensor is a set of named axes (no order) and
a dtype; sizes are expressions of variables.
    Variables   integers that fix the shapes once bound. External ones (`vars`) are bound by the
                parent through `bind`, as expressions of its own variables; internal ones of a
                composite (`internal`) are bound while running by `let`, from a child's output,
                which is how data-dependent sizes enter (e.g. rows per expert). An axis whose size
                is a tuple variable is ragged: its length differs for each index of the axis
                declared just before it.
    Leaves      ports, state and a `reference` in plain PyTorch on one device, with the bound
                variables in self.env: the specification of what the leaf computes.
    Composites  only connect children in `forward`; they compute nothing themselves, so what
                they connect depends on configuration alone and one run on handles traces them.
Whoever runs a model installs an interpreter (Module.interpreter), which receives every call of a
module and every let: the training plan's tracer, or Reference to run the references.

Public components
    var(names) -> tuple of sympy symbols
        Variables named in the space-separated `names`, e.g. B, S = var("batch seq").
    ModulePort(dtype, **axes)
        A port: axes by name with their sizes (ints or variable expressions), and a dtype.
    ModuleState(dtype, init="normal", trainable=True, sparse=None, **axes)
        State that outlives a call. init: normal, ones or zeros; trainable False: updated outside
        gradient descent (e.g. a router bias); sparse: an axis of which a call reads and updates
        only the indices an input names (e.g. the rows of an embedding table).
    Module(id, **bind)
        Base of all modules. id: unique among siblings, without dots; bind: the value of every
        variable in `vars`. Assigning a module (or a list of them) to an attribute makes it a
        child. inputs / outputs: ports by name (an input and an output of the same name are one
        tensor). global_id: the path; tensor_id(name): the global id of a port or state;
        resolve(tensor_id) -> (module, name), from the root, also through a module's
        `substitute` (an equivalent composite an implementation runs in its place); calling a
        module hands the call to the installed interpreter.
    LeafModule, .state, .reference(s, **inputs) -> dict of outputs
        s: the states as tensors, by name; inputs by port name. Tensors have their axes in
        declaration order, a ragged axis concatenated into the axis before it.
    CompositeModule, .internal, .forward(**inputs), .let(**values)
        forward calls children and returns an output or a dict of outputs; let binds internal
        variables to child outputs.
    Reference(states), Reference(states)(module, **inputs)
        Runs any module by its leaves' references: states maps global tensor ids to tensors (they
        may require gradients, so autograd gives reference gradients); a let binds the variable,
        as a tuple of the value's entries, in every module below the composite bound to it.
        call() and let() are its interpreter methods.
    CoarseLeaf(sources, **bind), .tensor_id(name), .run(s, module, **inputs)
        A leaf an implementation defines over modules of the model, to treat a large part of it
        as one unit. sources: original modules by a name of the leaf's choosing; their states
        become the leaf's, named "<source>.<state>" and keeping their original global ids, so the
        placement, copies and optimizer see the same tensors. Its ports are its own. Its reference
        runs the originals' references through run(s, module, **inputs), so it is correct by
        construction.

Example
    B, S = var("batch seq")

    class Scale(LeafModule):
        vars = ("batch", "seq")

        def __init__(self, dim, **bind):
            super().__init__(**bind)
            self.inputs = {"x": ModulePort(torch.bfloat16, batch=B, seq=S, dim=dim)}
            self.outputs = {"y": ModulePort(torch.bfloat16, batch=B, seq=S, dim=dim)}
            self.state = {"w": ModuleState(torch.float32, "ones", dim=dim)}

        def reference(self, s, x):
            return {"y": (x * s["w"]).to(x.dtype)}

    class Net(CompositeModule):
        vars = ("batch", "seq")

        def __init__(self, dim, **bind):
            super().__init__(**bind)
            self.a, self.b = Scale(dim, id="a", batch=B, seq=S), Scale(dim, id="b", batch=B, seq=S)
            self.inputs, self.outputs = dict(self.a.inputs), dict(self.b.outputs)

        def forward(self, x):
            return self.b(self.a(x))

    net = Net(8, id="net", batch=2, seq=16)
    net.tensor_id("x"), net.a.tensor_id("w")      # "net.x", "net.a.w"
    y = Reference({"net.a.w": torch.ones(8), "net.b.w": torch.ones(8)})(net, x=x)
"""
import sympy
import torch


def var(names: str):
    """Symbols for the variables named in the space-separated `names`."""
    out = sympy.symbols(names, integer=True, nonnegative=True)
    return out if isinstance(out, tuple) else (out,)


class ModulePort:
    def __init__(self, dtype: torch.dtype, **axes):
        self.dtype, self.axes = dtype, axes


class ModuleState(ModulePort):
    def __init__(self, dtype: torch.dtype, init: str = "normal", trainable: bool = True,
                 sparse: str | None = None, **axes):
        """init: normal, ones or zeros. trainable False: updated outside gradient descent.
        sparse: an axis of which a call reads and updates only the indices named by an input."""
        super().__init__(dtype, **axes)
        self.init, self.trainable, self.sparse = init, trainable, sparse


class Module:
    """Identity: every module has an `id`, given explicitly at construction and unique among its
    siblings, without dots. A module assigned to an attribute of another (directly, or in a list
    or tuple) is that module's child; its `global_id` is the parent's global id and its own id
    joined by a dot, the root's global id its id. Ports and states are identified by their keys
    in `inputs`, `outputs` and `state`; global_id(name) is the module's global id and the key
    joined by a dot, so an input and an output of the same name are one tensor."""

    vars: tuple[str, ...] = ()
    inputs: dict[str, ModulePort]
    outputs: dict[str, ModulePort]
    interpreter = None

    def __init__(self, id: str, **bind):
        assert id and "." not in id, (type(self).__name__, id)
        assert set(bind) == set(self.vars), (type(self).__name__, set(bind) ^ set(self.vars))
        object.__setattr__(self, "parent", None)
        object.__setattr__(self, "children", {})
        object.__setattr__(self, "substitute", None)  # an equivalent composite an implementation put in its place
        self.id, self.bind = id, bind

    def __setattr__(self, name, value):
        kids = [value] if isinstance(value, Module) else list(value) if isinstance(value, list | tuple) else []
        for child in kids:
            if isinstance(child, Module):
                assert child.parent is None, (child.id, "already has a parent")
                assert child.id not in self.children, (self.id, child.id, "duplicate id among siblings")
                object.__setattr__(child, "parent", self)
                self.children[child.id] = child
        object.__setattr__(self, name, value)

    @property
    def global_id(self) -> str:
        return f"{self.parent.global_id}.{self.id}" if self.parent is not None else self.id

    def resolve(self, tensor_id: str) -> tuple["Module", str]:
        """The module and port or state name of a tensor's global id, from this (root) module (also
        through a module's substitute, see training_plan.py)."""
        *path, name = tensor_id.split(".")
        assert path and path[0] == self.id, (self.id, tensor_id)
        m = self
        for part in path[1:]:
            m = m.children[part] if part in m.children else m.substitute.children[part]
        return m, name

    def tensor_id(self, name: str) -> str:
        """Global id of the input, output or state `name` of this module."""
        assert name in self.inputs or name in self.outputs or name in getattr(self, "state", {}), (self.global_id, name)
        return f"{self.global_id}.{name}"

    def __call__(self, *args, **kwargs):
        return Module.interpreter.call(self, *args, **kwargs)


class LeafModule(Module):
    state: dict[str, ModuleState]

    def reference(self, s: dict[str, torch.Tensor], **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class CompositeModule(Module):
    internal: tuple[str, ...] = ()

    def forward(self, **inputs):
        raise NotImplementedError

    def let(self, **values):
        """Binds internal variables to the values of child outputs."""
        assert set(values) <= set(self.internal), (type(self).__name__, set(values))
        Module.interpreter.let(self, **values)


class Reference:
    """Runs a module by its leaves' references on tensors: states maps global tensor ids to tensors.
    A let binds the variable, as a tuple of the value's entries, in every module below the composite
    whose binding is that variable."""

    def __init__(self, states: dict):
        self.states = states

    def __call__(self, module: Module, **inputs):
        previous, Module.interpreter = Module.interpreter, self
        try:
            return self.call(module, **inputs)
        finally:
            Module.interpreter = previous

    def call(self, module: Module, *args, **kwargs):
        if isinstance(module, CompositeModule):
            return module.forward(*args, **kwargs)
        given = dict(zip(module.inputs, args)) | {k: v for k, v in kwargs.items() if v is not None}
        st = {n: self.states[module.tensor_id(n)].to(s.dtype) for n, s in module.state.items()}
        out = module.reference(st, **given)
        return next(iter(out.values())) if len(out) == 1 else out

    def let(self, composite: CompositeModule, **values):
        def walk(m):
            for child in m.children.values():
                for k, e in child.bind.items():
                    if isinstance(e, sympy.Symbol) and e.name in values:
                        child.env[k] = tuple(values[e.name].tolist())
                walk(child)
        walk(composite)


class CoarseLeaf(LeafModule):
    """Coarse leaves: leaves an implementation defines over modules of the model, for the substitute of
    a composite (TrainingPlan.substitute), so that a large part of a model is one leaf with one
    implementation of few steps.

    A CoarseLeaf owns no state of its own: its states are those of the original modules it covers
    (`sources`, by a name of the leaf's choosing), named "<source>.<state>" and with the original
    tensors' global ids, so the placement, the copies and the optimizer see the same tensors. Its ports
    are its own. Its reference runs the original modules' references (self.run), so it is the model's
    specification by construction; an implementation of it is checked against that."""
    def __init__(self, sources: dict[str, Module], **bind):
        super().__init__(**bind)
        object.__setattr__(self, "sources", sources)  # not children: they stay in the model's tree
        self.state, self.origin = {}, {}
        for src, m in sources.items():
            for name, st in m.state.items():
                self.state[f"{src}.{name}"] = st
                self.origin[f"{src}.{name}"] = m.tensor_id(name)

    def tensor_id(self, name: str) -> str:
        """A state's id is the original tensor's; a port's is the leaf's own."""
        return self.origin[name] if name in self.origin else super().tensor_id(name)

    def run(self, s: dict, module: Module, **inputs):
        """The reference of an original module (leaf or composite) on the leaf's state tensors `s`."""
        return Reference({self.origin[k]: v for k, v in s.items()})(module, **inputs)
