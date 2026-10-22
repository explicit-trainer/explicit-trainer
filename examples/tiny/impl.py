"""The implementations of the example's leaves, as steps.

Purpose and design. Each side of each leaf is one compute step. The weight gradient is added into
the weight's "grad" copy by the backward step that computes it, so the micro-batches of a training
step accumulate there.

Public components
    LinearImpl
        forward(x) -> ({"y"}, saved); backward(saved, y) -> {"x"}, adding dw into w's "grad".
    SquaredErrorImpl
        forward(y, target) -> ({"loss"}, saved); backward(saved, loss) -> {"y"}.

Example
    impl = LinearImpl(module, copies, env, mesh, coord)
    outs, saved = impl.forward(x=x)                 # inside a Graph: records the step
"""
from explicit_trainer.framework.abstract.module_impl import ModuleImpl
from explicit_trainer.framework.abstract.step import step
from examples.tiny.model import Linear, SquaredError


@step
def _linear_forward(x, w, relu: bool):
    y = x @ w.t()
    return y.relu() if relu else y


@step(outputs=2)
def _linear_backward(x, w, y, dy, relu: bool):
    if relu:
        dy = dy * (y > 0)
    return dy @ w, dy.t() @ x


@step
def _squared_error(y, target):
    return (y - target).square().sum()


@step
def _squared_error_backward(y, target, dloss):
    return 2 * (y - target) * dloss


class LinearImpl(ModuleImpl):
    module_type = Linear

    def forward(self, x):
        w = self.read("w")
        y = _linear_forward(x, w, self.module.relu)
        return {"y": y}, {"x": x, "w": w, "y": y}

    def backward(self, saved, y):
        dx, dw = _linear_backward(saved["x"], saved["w"], saved["y"], y, self.module.relu)
        self.write("w", "grad", dw)
        return {"x": dx}


class SquaredErrorImpl(ModuleImpl):
    module_type = SquaredError

    def forward(self, y, target):
        return {"loss": _squared_error(y, target)}, {"y": y, "target": target}

    def backward(self, saved, loss):
        return {"y": _squared_error_backward(saved["y"], saved["target"], loss)}
