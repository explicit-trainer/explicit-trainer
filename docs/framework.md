# The framework

A training framework in which everything that happens in a training step is written down: the
model, where every tensor lives, every unit of work the GPU or the network does, and the order in
which it is launched. This file explains why it is built that way and what each abstraction is.

## Principles

**Explicit over automatic.** Writing code is cheap for agents, so we prefer spelling things out
to having the framework infer them. A layer's implementation lists its kernels in order instead of
relying on autograd; the placement lists every copy of every tensor instead of deriving them; the
plan lists every collective instead of leaving them to a wrapper. Nothing important happens that
cannot be found by reading the code, which is what makes a step easy to debug and to tune.

**Abstractions are schemas.** Each abstract class is a contract that concrete code fills in, and
each layer constrains the next. A module defines what its implementations must compute; an
implementation must be made only of local compute steps and communication steps; a schedule
decides only when and where steps are launched, never what the graph is. Because every model,
placement and plan is written against the same small set of classes, they look alike and fit
together.

**Built for profiling and debugging.** The unit of work is the step, a named function or
collective, so a profile is already grouped by meaning: time per step tells where the cost is
without untangling a kernel trace, which suits agents analysing it. Every module also carries a
plain PyTorch reference, so precision problems are found by comparing an implementation with its
reference on the same inputs, leaf by leaf or block by block.

## The pieces

In `src/explicit_trainer/framework/`:

```
abstract/module.py         the model: modules, ports, states, references, coarse leaves
abstract/placement.py      where every tensor lives: copies and their layouts
abstract/step.py           the vocabulary of work: steps, values, copy references
abstract/module_impl.py    how a module is computed, as steps
abstract/training_plan.py  one training step as one graph of steps
schedule.py                which stream each step runs on, in which order: a file per graph
trainer.py                 runs a plan as its schedule says: streams, the mesh, checks
```

`explicit_trainer.tools` reads profiles and validates, simulates and searches schedules offline.
The user's code provides the model, and for each way of running it a placement, module
implementations, a plan and a schedule generator (`examples/tiny/` is a complete small one).

### Module (abstract/module.py)

A model is a tree of modules with explicit ids; a tensor's global id is its module's path plus
its name (`net.layer-3.attn.wq.w`).

- `ModulePort` and `ModuleState`: a tensor declared by named axes (no order) and a dtype. A state
  outlives a call (weights, tables) and says how it is initialized and whether it is trained.
- Variables: integers that fix the shapes. External ones are bound by the parent; internal ones
  are bound while running by `let`, from a child's output, which is how data-dependent sizes
  (e.g. rows per expert) enter.
- `LeafModule`: ports, states and a `reference` in plain PyTorch on one device. The reference is
  the specification of what the leaf computes.
- `CompositeModule`: only connects children in `forward`; it computes nothing itself.
- `Reference`: runs any module by its leaves' references, for checking implementations.
- `CoarseLeaf`: a leaf an implementation defines over modules of the model. It owns no state of
  its own: its states are the original tensors (same ids), and its reference is the originals'
  references composed. It lets an implementation treat a large part of the model as one unit.

### Placement (abstract/placement.py)

`Placement.place(tensor_id)` returns every copy of that tensor, by name. A `Copy` has a `Layout`
(which axes are split over which mesh dimensions, which are partial sums, memory order, padding,
dtype, FP8 scale grouping, device or host memory), a lifetime (persistent, step, group, micro-batch,
call, backward), and the copy it is made from. A weight may rest sharded in FP32 ("value") and be read
whole in BF16 ("compute") or as FP8 copies; its gradient and optimizer state are copies too.
Implementations hold no form of a tensor that is not one of its copies.

### Step (abstract/step.py)

All work is steps, and a graph is ordinary code over `Value`s (handles, not data), recorded in a
`Graph` while it runs:

- compute step: a function marked `@step` (or `step(f, outputs=...)` when the outputs are known
  only at the call). It runs on one device, or on the host when its inputs are host values, and
  reads only its arguments.
- communication step: `all_gather`, `reduce_scatter`, `all_reduce`, `all_to_all` over mesh
  dimensions; `all_to_all` split sizes may be host values.
- `to_host`: a small tensor's value on the host, for sizes the next steps need.
- `input`: a tensor given from outside (tokens, labels, the step number).
- copies are accessed by steps: `read(tensor, copy)` gives a `CopyRef` argument; `write(tensor,
  copy, x)` attaches to the step that makes `x`. Accesses to one copy are ordered by program order.

The edges of the graph are exactly the Values a step reads and the order of accesses to copies.
The graph is static: data-dependent sizes are host values that later steps take as arguments.

### ModuleImpl (abstract/module_impl.py)

The implementation of one module (a leaf, or a composite as a whole) for the copies its placement
gives. `variables(**inputs)` computes extra sizes it needs (brought to the host by the plan and
bound in `env`); `forward(**inputs)` returns outputs and what the backward needs; `backward(saved,
**grads)` returns input gradients and writes state gradients into their copies. Both only build
steps. A good implementation has one compute step on each side of each communication.

### TrainingPlan (abstract/training_plan.py)

`build()` returns the graph of one whole training step on this device: making the step copies of
every state, every micro-batch's forward and backward through the module implementations, moving
values between layouts, summing gradients, reducing them and updating state. The base class traces
the model once to find the calls (`trace.ops`, the root's `trace.inputs` and `trace.outputs`). A
plan may replace a composite by an equivalent composite of coarse leaves (`substitute`, so the
graph is made of fewer, bigger steps), or implement a composite as a whole (`implements`). After
`build()` it holds `outputs` (what a training step returns, e.g. the loss) and `calls` (the Values
of every call, which `check()` compares).

### Schedule (schedule.py)

A schedule is data: `{"graph": <hash>, "queues": {<stream>: [step index, ...]}}` puts every step of
one graph on one stream (compute, to_host, host for steps on host values, or another named CUDA
stream, whose collectives get a process group of that name) and orders the steps of each stream.
It is generated offline (e.g. by a plan's generator) or written by hand, and checked against the
graph's `describe()` before use. Every device follows the same schedule, so collectives on each
stream are launched in the same order everywhere; only the interleaving of the streams depends on
timing. `explicit_trainer.tools.simulate` (the `et-schedule` command) simulates a step under a
schedule, with step times from a profile of any schedule of the same graph: step time, idle time
per stream and what it waited for, the critical path, a timeline.

### Trainer (trainer.py)

It launches the head of any queue of the schedule whose dependencies are launched and whose host
values have arrived, on its stream (a CUDA stream, or the CPU for host steps), with a CUDA event
wherever a step depends on one on another stream; it waits only when no head can be launched. It
holds every copy, initializes persistent ones the same way however they are split (a copy in host
memory page-locked and mapped, so that GPU kernels read and write it in place), and runs collectives
on process groups of a `Mesh`, whose NCCL communicators are all created and connected before the
first step (NCCL does both lazily, which can deadlock across machines in the middle of a step).
Work on the streams named `carry` may run on into the next training step; `step()` leaves the
compute stream behind all other work of the step. `step(serial=True)` waits for the device before
every launch, the streams taking turns, for the first step of a run: a kernel loaded on its first
launch (Triton, compiled functions) may wait for every kernel running on the device, while kernels
that wait for other ranks (e.g. NCCL) may be running, and the ranks deadlock. On a mesh without
CUDA (gloo, CPU tensors) every step runs synchronously on the CPU when it is launched. `check()`
runs a step (serially) next to the references and reports the relative error of the loss, of every
state gradient and, with `leaves=True`, of every implemented module's outputs and input gradients
on the plan's own inputs. An input split over devices is gathered whole first (in the order
`check(order=...)` gives, e.g. a sequence split in two chunks per device); an input every device
holds whole is used as it is.

What the trainer requires of a plan and its placement:

- `Mesh(dims)` needs `torch.distributed` initialized with as many ranks as the product of the
  dimensions; ranks are laid out row-major (for `{"node": 2, "gpu": 4}`, rank = node * 4 + gpu).
- A persistent copy may be split only along its state's first axis. The "value" copy starts as the
  state's `init` (normal with standard deviation 0.02, ones or zeros), drawn per block of rows from
  a seed of the tensor's id, so it is the same however the copy is split; other persistent copies
  (e.g. optimizer state) start as zeros.
- A collective acts along the first dimension of its tensor: an all-gather concatenates the
  members' tensors, a reduce-scatter gives member i the i-th block of the sum, an all-to-all sends
  and receives rows.
- A write is converted to the copy's dtype. With `accumulate=True` it adds to what the copy holds
  (it starts the copy if nothing does); with `accumulate=False` it replaces it. Only persistent
  copies outlive a training step; a copy of one micro-batch is freed after its last access.
