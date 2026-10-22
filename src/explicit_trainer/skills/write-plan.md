---
name: write-plan
description: Write or change a model, placement, module implementation or training plan with explicit-trainer. Use before adding a module, splitting a tensor differently, implementing a module as steps, or changing what one training step does.
---

# Writing a plan with explicit-trainer

Read `docs/framework.md` of the explicit-trainer repository first, and the header of each module named below: it
documents the interface in full. `examples/tiny/` of the repository is a complete small model, placement,
implementations, plan and schedule; read it before writing your own.

## The pieces

1. **Model** (`explicit_trainer.framework.abstract.module`): a tree of modules with explicit ids.
   - A `LeafModule` declares `inputs` and `outputs` (`ModulePort`: a dtype and named axes with their sizes),
     `state` (`ModuleState`: also `init`, `trainable`) and `reference(s, **inputs)`, plain PyTorch on one device that
     returns a dict of outputs. The reference is the specification of the leaf.
   - A `CompositeModule` only connects children in `forward`; data-dependent sizes enter through `let`.
   - A tensor's global id is the module path plus its name (`net.l1.w`: state `w` of child `l1` of root `net`).
2. **Placement** (`abstract/placement.py`): `place(tensor_id)` returns every `Copy` of a tensor by name. A `Copy` has
   a `Layout` (`split`: axis -> mesh dimensions; `partial`: mesh dimensions over which devices hold summands; `dtype`;
   `memory`), a lifetime (persistent, step, group, micro_batch, call, backward) and the copy it is made from. By
   convention every tensor has "value", and a trainable state "grad" and whatever its optimizer keeps.
3. **ModuleImpl** (`abstract/module_impl.py`): `forward(**inputs) -> (outputs, saved)` and
   `backward(saved, **grads) -> input grads` only record steps: functions marked `@step` and collectives
   (`all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all` of `abstract/step.py`). Read and write state copies
   with `self.read(name, copy)` and `self.write(name, copy, x, accumulate)`. Aim for one compute step on each side of
   each communication; inside a step anything local goes.
4. **TrainingPlan** (`abstract/training_plan.py`): `build()` returns the graph of one whole training step on this
   device, made inside `with Graph() as g:`: per micro-batch its `input(name, i)` steps and every call's forward
   and backward through its implementation (set `impl.micro_batch` first), moves between layouts, gradient sums and
   reductions, the optimizer. Set `self.outputs` (e.g. `{"loss": ...}`) and `self.calls` (`[module, inputs,
   outputs, output grads, input grads]` per call, Values by port name).

## What the trainer requires

- A persistent copy may be split only along its state's first axis: order a state's axes accordingly.
- A collective acts along the first dimension of its tensor.
- `Mesh(dims)` needs `torch.distributed` initialized with as many ranks as the product of the dimensions. On CPU,
  gloo works: `torchrun --nproc-per-node N` with `dist.init_process_group("gloo")`.

## Rules

- Every step is explicit: no autograd across steps, no implicit scheduling, no hidden collectives.
- The graph is static: a data-dependent size is a host value (`to_host`) passed to later steps as an argument.
- Never change the precision of a computation, storage or communication unless the task says so.
- After a change, make the graph's schedule (skill `write-schedule`) and compare with the references:
  `Trainer.check(micro_batches, grads, tokens, ...)` runs one step next to the model's references and reports the
  relative errors of the loss, of every state gradient and, with `leaves=True`, of every implemented call. Keep the
  errors at the levels the task documents.
