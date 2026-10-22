"""The training plan of the example: one training step as one graph.

Purpose and design. For every micro-batch: its inputs, every call's forward in model order, then
every call's backward in reverse order, seeded with d(loss) = 1 / (elements of the targets of the
whole step), so the gradients are those of the mean squared error. At the end of the step, every
weight's gradient (summed over the micro-batches in its "grad" copy) is all-reduced over "gpu" and
written back into "grad", and SGD with rate LR updates the weight. The loss output is the summed
squared error over all devices and micro-batches.

Public components
    LR
    TinyPlan(model, placement, coord, micro_batches)
        build() -> Graph; afterwards outputs = {"loss"} and calls.

Example
    plan = TinyPlan(net, DataParallel(net, 1), {"gpu": 0}, micro_batches=2)
    graph = plan.build()
"""
from explicit_trainer.framework.abstract.step import Graph, all_reduce, input, read, step, write
from explicit_trainer.framework.abstract.training_plan import TrainingPlan
from examples.tiny.impl import LinearImpl, SquaredErrorImpl

LR = 0.1
_IMPLS = (LinearImpl, SquaredErrorImpl)


@step
def _seed(loss, scale: float):
    return loss.new_full(loss.shape, scale)


@step
def _add(a, b):
    return a + b


@step
def _total(*losses):
    return sum(losses)


@step
def _gradient(g):
    return g.clone()


@step
def _sgd(w, g):
    return w - LR * g


class TinyPlan(TrainingPlan):
    def __init__(self, model, placement, coord, micro_batches):
        super().__init__(model, placement, coord, micro_batches)
        self.impls = {}
        for op in self.trace.ops:
            m = op.module
            copies = {n: placement.place(m.tensor_id(n)) for n in list(m.inputs) + list(m.outputs) + list(m.state)}
            impl = next(c for c in _IMPLS if c.accepts(m, copies))
            self.impls[m.global_id] = impl(m, copies, dict(m.env), placement.mesh, coord)
        self.weights = [op.module.tensor_id(n) for op in self.trace.ops for n in op.module.state]

    def build(self) -> Graph:
        target = self.model.inputs["target"].axes
        scale = 1.0 / (self.micro_batches * int(self.model.env["batch"]) * int(target["feature"]))
        loss_handle = self.trace.outputs["loss"]
        with Graph() as g:
            self.calls, losses = [], []
            for i in range(self.micro_batches):
                handles = {id(h): input(name, i) for name, h in self.trace.inputs.items()}
                tape = []
                for op in self.trace.ops:
                    impl = self.impls[op.module.global_id]
                    impl.micro_batch = i
                    args = {n: handles[id(h)] for n, h in op.inputs.items()}
                    outs, saved = impl.forward(**args)
                    for n, h in op.outputs.items():
                        handles[id(h)] = outs[n]
                    self.calls.append([op.module, args, outs, {}, {}])
                    tape.append((op, impl, saved, self.calls[-1]))
                losses.append(handles[id(loss_handle)])
                grads = {id(loss_handle): _seed(losses[-1], scale)}
                for op, impl, saved, call in reversed(tape):
                    out_grads = {n: grads.pop(id(h)) for n, h in op.outputs.items()}
                    back = impl.backward(saved, **out_grads)
                    call[3:] = [out_grads, back]
                    for n, gv in back.items():
                        h = op.inputs[n]
                        grads[id(h)] = _add(grads[id(h)], gv) if id(h) in grads else gv
            for w in self.weights:
                reduced = all_reduce(_gradient(read(w, "grad")), ("gpu",))
                write(w, "grad", reduced, accumulate=False)
                write(w, "value", _sgd(read(w, "value"), reduced), accumulate=False)
            self.outputs = {"loss": all_reduce(_total(*losses), ("gpu",))}
        return g
