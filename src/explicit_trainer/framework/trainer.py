"""The trainer: runs a training plan's graph, step after step, as its schedule says.

Purpose and design. A training step builds nothing: the plan's graph is built once, and each step
launches it as its schedule (schedule.py) says: one queue per stream, compute steps on the compute
stream (or another named one), to_host copies on a stream of their own, host steps on the CPU,
collectives on a named stream with a process group of that name, so that streams overlap. The
trainer launches the head of any queue whose dependencies are launched and whose host values have
arrived, and when no head can go, waits (polling) for the first host value a head needs. A step
waits on the device (a CUDA event) for the steps it depends on that ran on other device streams.
The trainer holds every copy the graph accesses: persistent copies across steps (a state's "value"
initialized from its declaration, the others zeros of its shape; where the copy's layout says
memory "host", in page-locked host memory mapped for the device, which GPU kernels read and write in
place), the others from their first write to the step's end, or for copies of one micro-batch to
their last access. A write is in the copy's dtype; with `accumulate` it adds to what the copy holds
(a pair of row indices and rows, a sparse gradient, is concatenated). A Value is freed once its last
reader has been launched; a copy that lives shorter than the step, once its last access has been
launched, or its last access before a write replaces it whole (a generation). A collective acts on
the first dimension of its tensor; FP8 tensors travel as bytes. Initial values are made per block
of 4096 rows of a state's first axis from a seed of (tensor id, block), so a state starts the same
however it is split, on any mesh; a persistent copy may therefore be split only along that axis.
Without CUDA (a gloo mesh of CPU processes) there are no streams: every step runs on the CPU when
it is launched.

Checking. check() runs one training step (serially) next to the model's references (module.py,
Reference) on the first device, from the same initial state and on the whole batch (every input
gathered along the axis its placement splits, the batch or the sequence, over the mesh dimensions
it is split over; an input that every device holds whole is taken as it is): the loss, every state
gradient, and with leaves=True every call of every implemented module on the inputs the plan gave
it (outputs, input gradients with their projection onto the reference, and the per-call sum of
state gradients). On a mesh of more than one device, calls with a port that has one of the `local`
axes (rows a plan moved between devices, which no device holds whole) are left out.

Public components
    Mesh(dims), .dims, .names, .rank, .world, .coord, .cuda, .device, .groups
        Ranks laid out over the mesh dimensions in row-major order (for {"node": 2, "gpu": 4},
        rank = node * 4 + gpu); coord: this rank's index along each dimension. Needs
        torch.distributed initialized.
    Mesh.add(groups)
        groups: (mesh dimensions, name) pairs; a process group for each, named so that collectives of
        different names never share a communicator (every rank creates every group in the same
        order); groups[(dims, name)]. Each NCCL communicator is created and its members connected
        right away, one at a time (a tiny all-reduce and a 4 MiB-per-member all-to-all each): done
        lazily during a step, either can deadlock across machines.
    Mesh.span(dims), Mesh.index(dims), Mesh.size(dims), Mesh.members(dims)
        The mesh dimensions among dims in mesh order; this rank's position in its group spanning
        them; the group's size; the ranks of the group spanning them at index 0 of every other
        dimension, in group order.
    Trainer(plan, schedule, mesh, carry=()), .graph, .outputs, .copies, .annotate
        schedule: a schedule of the plan's graph (schedule.py), checked against it. Builds the graph
        once, makes a process group per stream of collectives and set of mesh dimensions its
        collectives span, and initializes the persistent copies. carry: streams whose work of a
        step may run on into the next step (the next step's work on them follows it). annotate:
        name every step in profiler traces, as "step <kind> #<index in graph.steps>" (schedule.kind).
    Trainer.step(micro_batches, serial=False, **inputs) -> dict
        One training step: micro_batches is this device's inputs of every micro-batch (dicts of
        tensors by root input name); inputs are inputs of the whole step (e.g. step=1). Returns
        the plan's outputs once every step is launched; the compute stream then has all the step's
        work behind it but that of the carry streams, so torch.cuda.current_stream().synchronize()
        waits for the step. serial: wait on the host for every step before launching the next, for
        the first step of a run: a kernel is loaded on its first launch (Triton, compiled
        functions), and loading may wait for every kernel running on the device, while a kernel
        that waits for the other ranks (e.g. NCCL) may be running, whose peers wait for this
        rank: the ranks deadlock. Run serially, every launch finds the device idle; the streams take
        turns, one step each, so that none runs far ahead of the others (holding more memory).
    Trainer.check(micro_batches, grads, tokens, bind=(), local=(), leaves=False, whole=True, order=None,
                  **inputs) -> dict
        grads: per trainable state, the copy holding its gradient at the end of the step, in the
        layout of its value (or as a pair of local row indices and rows). tokens: a root input (e.g.
        "labels"); the reference's gradients of the loss summed over the micro-batches are divided
        by its number of elements in the whole batch, so the plan's gradients must be those of the
        mean loss per element. bind: output ports whose values bind the internal variable of the
        same name in the calls that follow (e.g. rows per expert), when the leaves are compared.
        local: axes of rows moved between devices; on a mesh of more than one device, calls with a
        port that has one are left out. order: the chunks of a whole input in order, as (device,
        chunk) pairs, each device's part cut into equal chunks (e.g. a sequence split in two chunks
        per device; device: the position in the group the input is split over); default each
        device's part, in order. The step checked is a training step: the plan's optimizer updates
        the state. Returns, on rank 0 (others return {}), "loss": (plan's, reference's) and relative
        errors by key: state ids, "leaf <type>.<port>", "calls <state>", "scale ...".
        whole=False leaves out the whole model's reference (loss and state gradients), which needs
        the whole model in FP32 several times on the first device; only the leaves are compared.

Example
    dist.init_process_group("nccl")
    mesh = Mesh({"node": 1, "gpu": 4})
    plan = SomePlan(model, placement, mesh.coord, micro_batches=8)
    trainer = Trainer(plan, json.load(open("schedule.json")), mesh)
    for step in range(1, 101):
        out = trainer.step(next_batches(), step=step)
        print(out["loss"].item())
"""
import contextlib
import hashlib
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist

from explicit_trainer.framework.abstract.module import ModuleState
from explicit_trainer.framework.abstract.step import AllGather, AllReduce, AllToAll, CommunicationStep, \
    ComputeStep, CopyRef, Input, ReduceScatter, ToHost, Value, Write
from explicit_trainer.framework.abstract.training_plan import Reference, TrainingPlan
from explicit_trainer.framework.schedule import DEVICE_ONLY, check, describe, kind


class Mesh:
    def __init__(self, dims: dict[str, int]):
        self.dims, self.names = dims, tuple(dims)
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        assert self.world == torch.Size(dims.values()).numel(), (dims, self.world)
        self.coord = dict(zip(self.names, self._unravel(self.rank)))
        self.groups = {}
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda", torch.cuda.current_device()) if self.cuda else torch.device("cpu")

    def add(self, groups: list[tuple[tuple[str, ...], str]]):
        """A process group per (mesh dimensions, name), only those asked for: each holds buffers."""
        for span, name in groups:
            for members in self._partition(span):
                g = dist.new_group(members)  # every rank creates every group, in the same order
                if self.rank in members:
                    self.groups[span, name] = g
        # NCCL creates a communicator on its first collective, and connects two members on their first
        # send and receive (an all-to-all), blocking the host until the peer takes part; done during a
        # step, where ranks may reach two communicators in different orders, either deadlocks across
        # machines. So every communicator is created and connected now, one at a time, in the same
        # order everywhere.
        for g in self.groups.values():
            dist.all_reduce(torch.zeros(1, device=self.device), group=g)
            n = dist.get_world_size(g) * _CONNECT  # large enough to use every channel to every member
            dist.all_to_all_single(torch.zeros(n, device=self.device), torch.zeros(n, device=self.device), group=g)
            if self.cuda:
                torch.cuda.synchronize()

    def _unravel(self, rank: int) -> list[int]:
        out = []
        for size in reversed(list(self.dims.values())):
            out.append(rank % size)
            rank //= size
        return out[::-1]

    def _partition(self, span: tuple[str, ...]) -> list[list[int]]:
        """The rank lists of the groups spanning `span`: ranks that agree on every other dimension."""
        keep = [i for i, n in enumerate(self.names) if n not in span]
        parts = {}
        for r in range(self.world):
            c = self._unravel(r)
            parts.setdefault(tuple(c[i] for i in keep), []).append(r)
        return list(parts.values())

    def span(self, dims: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(n for n in self.names if n in dims)

    def index(self, dims: tuple[str, ...]) -> int:
        """This rank's position in its group spanning `dims`."""
        i = 0
        for d in self.span(dims):
            i = i * self.dims[d] + self.coord[d]
        return i

    def size(self, dims: tuple[str, ...]) -> int:
        n = 1
        for d in self.span(dims):
            n *= self.dims[d]
        return n

    def members(self, dims: tuple[str, ...]) -> list[int]:
        """The ranks of the group spanning `dims` that holds the first index of every other dimension, in group
        order."""
        span = self.span(dims)
        return [r for r in range(self.world)
                if all(c == 0 for n, c in zip(self.names, self._unravel(r)) if n not in span)]


_BLOCK = 4096
_CONNECT = 1 << 20  # values sent to each member when a communicator is connected
_INIT_THREADS = 16
_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)


def _init_rows(state: ModuleState, key: str, lo: int, hi: int, std: float = 0.02) -> torch.Tensor:
    """Rows lo .. hi of the first axis of a state's initial value, in FP32 on the host."""
    shape = tuple(int(v) for v in state.axes.values())
    rest = shape[1:]
    if state.init != "normal":
        return torch.full((hi - lo, *rest), 1.0 if state.init == "ones" else 0.0)
    out = []
    for b in range(lo // _BLOCK, -(-hi // _BLOCK)):
        seed = int.from_bytes(hashlib.sha256(f"{key}#{b}".encode()).digest()[:8], "little") >> 1
        n = min(_BLOCK, shape[0] - b * _BLOCK)
        block = torch.randn(n, *rest, generator=torch.Generator().manual_seed(seed)) * std
        out.append(block[max(lo - b * _BLOCK, 0): hi - b * _BLOCK])
    return torch.cat(out) if out else torch.empty(0, *rest)


def _pinned(state: ModuleState, key: str, lo: int, hi: int, dtype: torch.dtype, initial: bool) -> torch.Tensor:
    """Rows lo .. hi of a state as a host tensor that GPU kernels read and write in place: filled by
    _INIT_THREADS threads (the initial value, the same as _init_rows makes, or zeros), then
    page-locked and mapped into the device's address space (cudaHostRegister: exactly its size,
    where pinned allocations round up to a power of two)."""
    x = torch.empty(hi - lo, *(int(v) for v in list(state.axes.values())[1:]), dtype=dtype)

    def fill(a: int):
        b = min(a + _BLOCK * 16, hi)
        x[a - lo: b - lo] = _init_rows(state, key, a, b) if initial else 0

    with ThreadPoolExecutor(_INIT_THREADS) as pool:  # a large table: its blocks in parallel
        list(pool.map(fill, range(lo, hi, _BLOCK * 16)))
    err = torch.cuda.cudart().cudaHostRegister(x.data_ptr(), x.numel() * x.element_size(), 3)  # portable, mapped
    assert int(err) == 0, (key, err)
    return x


class _HostValue:
    """The value of a device tensor on the host, copied asynchronously; get() waits for it."""

    def __init__(self, x: torch.Tensor):
        self.host = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
        self.host.copy_(x, non_blocking=True)
        self.done = torch.cuda.Event()
        self.done.record()

    def get(self) -> list:
        self.done.synchronize()
        return self.host.tolist()


def _as_bytes(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.uint8) if x.dtype in _FP8 else x


class Trainer:
    def __init__(self, plan: TrainingPlan, schedule: dict, mesh: Mesh, carry: tuple[str, ...] = ()):
        self.plan, self.mesh, self.device, self.carry = plan, mesh, mesh.device, carry
        self.graph = plan.build()
        info = describe(self.graph)
        check(schedule, info)
        steps = self.graph.steps
        self.queues = {q: [steps[i] for i in members] for q, members in schedule["queues"].items()}
        self.index = {s: i for i, s in enumerate(steps)}
        self.stream = {s: q for q, members in self.queues.items() for s in members}
        mesh.add(sorted({(mesh.span(s.dims), q) for s, q in self.stream.items() if isinstance(s, CommunicationStep)}))
        self.outputs = plan.outputs
        self.streams = {q: torch.cuda.current_stream() if q == "compute" else torch.cuda.Stream()
                        for q in set(self.queues) - set(DEVICE_ONLY) | {"to_host"}} if mesh.cuda else {}
        self.deps = {s: [steps[d] for d in info["steps"][i]["deps"]] for i, s in enumerate(steps)}
        self.arrivals = {s: [d for d in self.deps[s] if isinstance(d, ToHost)] for s in steps}
        # the steps each step waits for on the device: its dependencies on other device streams
        self.waits = {s: [d for d in self.deps[s] if self.stream[d] not in ("host", self.stream[s])] for s in steps}
        self.signals = {d for w in self.waits.values() for d in w}
        # the accesses of each step to copies shorter than the step, by generation: a write that
        # replaces such a copy whole begins a new one; each generation is freed after its last access
        self.accesses, self.generations, generation = {}, {}, {}
        for s in steps:  # in program order; a step's reads come before its writes
            self.accesses[s] = [(c.key, generation.setdefault(c.key, 0)) for c in s.reads()
                                if c.micro_batch is not None]
            self.generations[s] = {}
            for w in s.writes:
                if w.copy.micro_batch is not None:
                    k = w.copy.key
                    generation[k] = generation[k] + 1 if k in generation and not w.accumulate else generation.get(k, 0)
                    self.generations[s][k] = generation[k]
                    self.accesses[s].append((k, generation[k]))
        # how many steps read each Value and access each generation of a copy: freed after the last one
        self.readers = Counter(k for s in steps for k in [(v.step, v.index) for v in s.values()] + self.accesses[s])
        self.kept = {(v.step, v.index) for v in self.outputs.values()}
        self.copies = {}
        refs = [c for s in self.graph.steps for c in s.reads()] + [w.copy for s in self.graph.steps for w in s.writes]
        for tensor in {c.tensor: None for c in refs}:
            for c in plan.placement.place(tensor).values():
                if c.lifetime == "persistent":
                    self.copies[tensor, c.name, None] = self._initial(tensor, c)
        self.persistent = set(self.copies)
        self.keep, self.kept_copies = set(), {}  # copies to keep at the end of a step (check)
        self.held, self.values = set(), {}  # Values never freed, and the last step's (check)
        self.annotate = False  # name every step in profiler traces
        self.copy_stream = {}  # copy key -> the stream of its last write

    # ------------------------------------------------------------ copies

    def _initial(self, tensor: str, copy) -> torch.Tensor:
        module, name = self.plan.model.resolve(tensor)
        state = module.state[name]
        axes = list(state.axes)
        n = int(state.axes[axes[0]])
        lo, hi = 0, n
        for axis, dims in copy.layout.split.items():
            assert axis == axes[0], (tensor, "split only along the first axis")
            k = self.mesh.size(dims)
            rows = -(-n // k)
            i = self.mesh.index(dims)
            lo, hi = min(i * rows, n), min((i + 1) * rows, n)
        dtype, rest = copy.layout.dtype or state.dtype, [int(state.axes[a]) for a in axes[1:]]
        if copy.layout.memory == "host" and self.mesh.cuda:
            return _pinned(state, tensor, lo, hi, dtype, copy.name == "value")
        x = _init_rows(state, tensor, lo, hi) if copy.name == "value" else torch.zeros(hi - lo, *rest)
        return x.to(self.device, dtype)

    def _dtype(self, ref: CopyRef):
        module, name = self.plan.model.resolve(ref.tensor)
        c = self.plan.placement.place(ref.tensor)[ref.copy]
        declared = (module.state.get(name) if hasattr(module, "state") else None) or \
            module.inputs.get(name) or module.outputs.get(name)
        return c.layout.dtype or declared.dtype

    def _write(self, w: Write, x):
        key = w.copy.key
        if isinstance(x, torch.Tensor):
            dtype = self._dtype(w.copy)
            if dtype not in _FP8:
                x = x.to(dtype)
        if not w.accumulate or key not in self.copies:
            self.copies[key] = x
        elif isinstance(x, tuple):
            self.copies[key] = tuple(torch.cat([a, b]) for a, b in zip(self.copies[key], x))
        else:
            self.copies[key].add_(x)

    # ------------------------------------------------------------ the step

    def step(self, micro_batches: list[dict[str, torch.Tensor]], serial: bool = False, **inputs) -> dict:
        """Launches every step of the graph as the schedule says (with serial, each once the device has
        finished the one before). Returns the plan's outputs."""
        self.given = {(k, i): x.pin_memory().to(self.device, non_blocking=True) if self.mesh.cuda else x
                      for i, mb in enumerate(micro_batches) for k, x in mb.items()}
        self.given |= {(k, None): x for k, x in inputs.items()}
        self.values, self.done, self.left, self.written = {}, {}, Counter(self.readers), {}
        launched, pending = set(), {q: deque(members) for q, members in self.queues.items()}
        while any(pending.values()):
            moved = False
            for q, queue in pending.items():
                while queue and all(d in launched for d in self.deps[queue[0]]) and \
                        all(self._arrived(d) for d in self.arrivals[queue[0]]):
                    s = queue.popleft()
                    if serial and self.mesh.cuda:
                        torch.cuda.synchronize()
                    self._launch(s, q)
                    launched.add(s)
                    moved = True
                    if serial:  # one step per stream in turn, so that the streams advance together
                        break
            if not moved:
                needed = [d for queue in pending.values() if queue and all(d in launched for d in self.deps[queue[0]])
                          for d in self.arrivals[queue[0]]]
                assert needed, "no queue can go and none waits for a host value"
                while not any(self._arrived(d) for d in needed):
                    time.sleep(0)
        for q, s in self.streams.items():
            if q not in self.carry:
                torch.cuda.current_stream().wait_stream(s)
        result = {k: self.values[v.step, v.index] for k, v in self.outputs.items()}
        self.values = {k: self.values[k] for k in self.held}
        self.kept_copies = {(t, c): x for (t, c, _), x in self.copies.items() if (t, c) in self.keep}
        self.copies = {k: v for k, v in self.copies.items() if k in self.persistent}
        return result

    def _arrived(self, s) -> bool:
        x = self.values[s, 0]
        return not isinstance(x, _HostValue) or x.done.query()

    def _launch(self, s, q: str):
        """Runs the step on its stream, then its writes there (a write is part of its step)."""
        cuda = self.streams.get(q)
        if cuda is not None:
            for d in self.waits[s]:
                cuda.wait_event(self.done[d])
        with torch.cuda.stream(cuda) if cuda is not None else contextlib.nullcontext():
            out = self._run(s, q)
            if s.outputs:
                for i, y in enumerate(out if s.outputs > 1 else (out,)):
                    self.values[s, i] = y
            for w in s.writes:
                self._write(w, tuple(self.values[v.step, v.index] for v in w.x) if isinstance(w.x, tuple)
                            else self.values[w.x.step, w.x.index])
                self.copy_stream[w.copy.key] = q
            if cuda is not None and s in self.signals:
                self.done[s] = torch.cuda.Event()
                self.done[s].record(cuda)
        self.written |= self.generations[s]  # the generation each copy holds
        for k in [(v.step, v.index) for v in s.values()] + self.accesses[s]:
            self.left[k] -= 1
            if not self.left[k]:
                self._free(k)
        for i in range(s.outputs):
            if not self.readers[s, i]:
                self._free((s, i))

    def _free(self, k: tuple):
        """Releases a Value (unless kept or held), or a generation (key, generation) of a copy
        shorter than the step if the copy still holds it."""
        if isinstance(k[0], tuple):
            if self.written.get(k[0]) == k[1]:
                self.copies.pop(k[0], None)
        elif k not in self.kept and k not in self.held:
            self.values.pop(k, None)

    # ------------------------------------------------------------ checking

    def _whole(self, tensor: str, x: torch.Tensor) -> torch.Tensor:
        """A tensor in the layout of the state's value, gathered whole (padded blocks trimmed)."""
        layout = self.plan.placement.place(tensor)["value"].layout
        x = x.to(self.device)
        module, name = self.plan.model.resolve(tensor)
        n = int(next(iter(module.state[name].axes.values())))
        for dims in layout.split.values():
            k = self.mesh.size(dims)
            rows = -(-n // k)
            i = self.mesh.index(dims)
            x = x[: max(0, min((i + 1) * rows, n) - min(i * rows, n))]  # the block (a copy may hold it padded)
            x = torch.cat([x, x.new_zeros(rows - x.size(0), *x.shape[1:])]).contiguous()
            out = x.new_empty(k * rows, *x.shape[1:])
            group = next(g for (span, _), g in self.mesh.groups.items() if span == self.mesh.span(dims))
            dist.all_gather_into_tensor(out, x, group=group)
            x = out[:n]
        return x

    def _state(self, tensor: str) -> ModuleState:
        module, name = self.plan.model.resolve(tensor)
        return module.state[name]

    def _reference(self, states: dict, micro_batches: list[dict]):
        """The summed loss of the references over the micro-batches, and the states' gradients. The
        states are used in place (not copied): a whole model in FP32 is large."""
        leaves = {n: x.detach().requires_grad_(self._state(n).trainable) for n, x in states.items()}
        trainable = [n for n in leaves if self._state(n).trainable]
        loss, grads = 0.0, {}
        for mb in micro_batches:
            with torch.enable_grad():
                part = Reference(leaves)(self.plan.model, **mb)
            got = torch.autograd.grad(part, [leaves[n] for n in trainable], allow_unused=True)
            for n, g in zip(trainable, got):
                if g is not None:
                    grads[n] = grads[n] + g if n in grads else g
            loss = loss + part.detach()
        return loss, {n: grads[n] if n in grads else torch.zeros_like(leaves[n]) for n in trainable}

    def _check_leaves(self, states: dict, bind: tuple[str, ...], local: tuple[str, ...]) -> dict:
        """Errors per implemented module type (a leaf, or a composite against its children's
        references); also sums, per state, the references' gradients of every call (self.state_sums,
        on one device only), to compare with the plan's without the drift of the forward. With more
        than one device, calls with a port that has one of the `local` axes are left out."""
        worst, self.state_sums = {}, {}

        def get(v):
            return self.values[v.step, v.index]

        def error(got, r):
            if not r.dtype.is_floating_point:
                return (got != r).float().mean().item()
            return ((got.float() - r.float()).norm() / r.float().norm().clamp_min(1e-30)).item()

        bound = {}  # internal variables as the calls' outputs `bind` made them, by name
        for m, ins, outs, dout, din in self.plan.calls:
            ports = list(m.inputs.values()) + list(m.outputs.values())
            if self.mesh.world > 1 and any(a in p.axes for p in ports for a in local):
                continue  # rows moved between devices: the reference sees them whole on one device only
            for k, v in outs.items():
                if k in bind:
                    bound[k] = tuple(get(v).tolist())
            m.env = dict(m.env) | {k: bound[k] for k in m.vars if m.env[k] is None and k in bound}
            ids = {m.tensor_id(n) for n in m.state}  # a coarse leaf's are the original tensors
            leaves = {t: states[t].detach().requires_grad_(self._state(t).trainable) for t in ids}
            x = {k: get(v).detach().requires_grad_(get(v).dtype.is_floating_point) for k, v in ins.items()}
            with torch.enable_grad():
                ref = Reference(leaves)(m, **x)
            if not isinstance(ref, dict):
                ref = {next(iter(m.outputs)): ref}
            found = {k: error(get(v), ref[k]) for k, v in outs.items()}
            wrt = [k for k in din if x[k].requires_grad]
            trained = [t for t, y in leaves.items() if y.requires_grad]
            ys = [k for k in dout if ref[k].requires_grad]
            if ys and (wrt or trained):
                got = torch.autograd.grad([ref[k] for k in ys], [x[k] for k in wrt] + [leaves[t] for t in trained],
                                          [get(dout[k]) for k in ys], allow_unused=True)
                found |= {f"d{k}": error(get(din[k]), g if g is not None else torch.zeros_like(x[k]))
                          for k, g in zip(wrt, got)}
                for k, g in zip(wrt, got):  # projection onto the reference: 1 when the error is only noise
                    if g is not None:
                        a, r = get(din[k]).float(), g.float()
                        found[f"scale d{k}"] = ((a * r).sum() / r.square().sum().clamp_min(1e-30)).item()
                for t, g in zip(trained, got[len(wrt):]):
                    if g is not None and self.mesh.world == 1:  # whole on one device only
                        self.state_sums[t] = self.state_sums.get(t, 0) + g.float()
            for k, e in found.items():
                key = f"leaf {type(m).__name__}.{k}"
                if k.startswith("scale "):  # the one farthest from 1, and every call's
                    worst[f"call {m.global_id}.{k}"] = e
                    worst[key] = e if key not in worst or abs(e - 1) > abs(worst[key] - 1) else worst[key]
                    continue
                else:
                    worst[key] = max(worst.get(key, 0.0), e)
        return worst

    def check(self, micro_batches: list[dict[str, torch.Tensor]], grads: dict[str, str], tokens: str,
              bind: tuple[str, ...] = (), local: tuple[str, ...] = (), leaves: bool = False, whole: bool = True,
              order: list[tuple[int, int]] | None = None, **inputs) -> dict:
        states = {t: self._whole(t, x.float()) for (t, c, _), x in self.copies.items() if c == "value"}
        mbs = []
        for mb in micro_batches:
            gathered = {}
            for k, x in mb.items():  # whole, along the axis the placement splits the input over
                tid = self.plan.trace.inputs[k].tensor
                module, name = self.plan.model.resolve(tid)
                split = self.plan.placement.place(tid)["value"].layout.split
                if not split:  # every device holds it whole
                    gathered[k] = x.to(self.device)
                    continue
                (axis, dims), = split.items()
                dim = list(module.inputs[name].axes).index(axis)
                parts = [torch.empty_like(x, device=self.device) for _ in range(self.mesh.world)]
                dist.all_gather(parts, x.to(self.device).contiguous())
                parts = [parts[r] for r in self.mesh.members(dims)]
                if order is not None:
                    chunks = [p.chunk(len(order) // len(parts), dim) for p in parts]
                    parts = [chunks[d][j] for d, j in order]
                gathered[k] = torch.cat(parts, dim)
            mbs.append(gathered)
        self.keep = set(grads.items())
        self.held = {(v.step, v.index) for call in self.plan.calls for port in call[1:] for v in port.values()} \
            if leaves else set()
        result = self.step(micro_batches, serial=True, **inputs)
        self.state_sums = {}  # per state, the references' gradients summed over the calls (_check_leaves)
        per_leaf = self._check_leaves(states, bind, local) if leaves else {}
        self.held, self.values = set(), {}  # the calls' values are compared: free them for the reference
        if not whole:
            return {"loss": (result["loss"].item(), None)} | per_leaf if self.mesh.rank == 0 else {}
        got = {}
        for t, c in grads.items():
            g = self.kept_copies[t, c]
            if isinstance(g, tuple):  # local row indices and rows
                index, rows = g
                shard = self.copies[t, "value", None]
                g = torch.zeros(shard.shape, dtype=torch.float32, device=self.device).index_add_(0, index, rows.float())
            got[t] = self._whole(t, g.float())
        self.keep = set()
        if self.mesh.rank != 0:
            return {}
        count = sum(mb[tokens].numel() for mb in mbs)
        loss, ref = self._reference(states, mbs)
        report = {"loss": (result["loss"].item(), loss.item())} | per_leaf
        for t, g in got.items():
            r = ref[t].float() / count
            report[t] = ((g - r).norm() / r.norm().clamp_min(1e-30)).item()
            if t in self.state_sums and self.mesh.world == 1:  # per-call sums are whole on one device only
                r = self.state_sums[t]
                report["calls " + t] = ((g - r).norm() / r.norm().clamp_min(1e-30)).item()
                report["scale " + t] = ((g * r).sum() / r.square().sum().clamp_min(1e-30)).item()  # 1 if only noise
        return report

    def _resolve(self, a, stream):
        """The data of an argument; waits on the host for a host value. A tensor made (or a copy
        written) on another device stream is marked as used on this one, so that freeing it waits
        for this use."""
        if isinstance(a, tuple):
            return tuple(self._resolve(x, stream) for x in a)
        if isinstance(a, CopyRef):
            x = self.copies[a.key]
            if stream in self.streams and self.copy_stream.get(a.key, stream) != stream:
                for t in x if isinstance(x, tuple) else (x,):
                    if isinstance(t, torch.Tensor) and t.is_cuda:
                        t.record_stream(self.streams[stream])
            return x
        if not isinstance(a, Value):
            return a
        x = self.values[a.step, a.index]
        if isinstance(x, _HostValue):
            x = self.values[a.step, a.index] = x.get()
        if stream in self.streams and isinstance(x, torch.Tensor) and x.is_cuda and self.stream[a.step] != stream:
            x.record_stream(self.streams[stream])
        return x

    def _run(self, s, stream):
        args = tuple(self._resolve(a, stream) for a in s.args)
        if self.annotate and not isinstance(s, Input):
            with torch.profiler.record_function(f"step {kind(s)} #{self.index[s]}"):
                return self._call(s, args, stream)
        return self._call(s, args, stream)

    def _call(self, s, args, stream):
        if isinstance(s, ComputeStep):
            return s.fn(*args)
        if isinstance(s, Input):
            return self.given[s.name, s.micro_batch]
        if isinstance(s, ToHost):
            return _HostValue(args[0]) if self.mesh.cuda else args[0].tolist()
        return self._collective(s, stream, args)

    def _collective(self, s, stream, args):
        group = self.mesh.groups.get((self.mesh.span(s.dims), stream))
        x = args[0]
        if group is None:  # a group of one device
            return x
        n, y = self.mesh.size(s.dims), _as_bytes(x.contiguous())
        if isinstance(s, AllGather):
            out = y.new_empty(n * y.size(0), *y.shape[1:])
            dist.all_gather_into_tensor(out, y, group=group)
        elif isinstance(s, ReduceScatter):
            out = y.new_empty(y.size(0) // n, *y.shape[1:])
            dist.reduce_scatter_tensor(out, y, group=group)
        elif isinstance(s, AllReduce):
            out = y.clone()
            dist.all_reduce(out, op=dist.ReduceOp.MAX if s.op == "max" else dist.ReduceOp.SUM, group=group)
        elif isinstance(s, AllToAll):
            send, recv = list(args[1]), list(args[2])
            y = y[: sum(send)]  # rows after those sent (padding) are not part of the exchange
            out = y.new_empty(sum(recv), *y.shape[1:])
            dist.all_to_all_single(out, y, recv, send, group=group)
        else:
            raise TypeError(s)
        return out.view(x.dtype) if x.dtype in _FP8 else out
