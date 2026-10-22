"""The model of the example: x -> Linear (ReLU) -> Linear -> squared error against a target.

Purpose and design. Two leaves with plain PyTorch references and one composite that connects them.
The batch size is the variable "batch" (the whole batch over all devices); feature sizes are fixed
at construction.

Public components
    B
        The variable "batch".
    Linear(d_in, d_out, relu, id, batch)
        Input x [batch, d_in], output y [batch, d_out], state w [d_out, d_in]: y = x @ w.T (then ReLU).
    SquaredError(d, id, batch)
        Inputs y and target [batch, d], output loss (a scalar): the sum of squared differences.
    Net(d_in, d_hidden, d_out, id, batch)
        Inputs x and target, output loss.

Example
    net = Net(16, 32, 8, id="net", batch=64)
    loss = Reference(states)(net, x=x, target=target)
"""
import torch

from explicit_trainer.framework.abstract.module import CompositeModule, LeafModule, ModulePort, ModuleState, var

(B,) = var("batch")


class Linear(LeafModule):
    vars = ("batch",)

    def __init__(self, d_in: int, d_out: int, relu: bool, **bind):
        super().__init__(**bind)
        self.relu = relu
        self.inputs = {"x": ModulePort(torch.float32, batch=B, feature=d_in)}
        self.outputs = {"y": ModulePort(torch.float32, batch=B, feature=d_out)}
        self.state = {"w": ModuleState(torch.float32, rows=d_out, cols=d_in)}

    def reference(self, s, x):
        y = x @ s["w"].t()
        return {"y": y.relu() if self.relu else y}


class SquaredError(LeafModule):
    vars = ("batch",)

    def __init__(self, d: int, **bind):
        super().__init__(**bind)
        self.inputs = {"y": ModulePort(torch.float32, batch=B, feature=d),
                       "target": ModulePort(torch.float32, batch=B, feature=d)}
        self.outputs = {"loss": ModulePort(torch.float32)}
        self.state = {}

    def reference(self, s, y, target):
        return {"loss": (y - target).square().sum()}


class Net(CompositeModule):
    vars = ("batch",)

    def __init__(self, d_in: int, d_hidden: int, d_out: int, **bind):
        super().__init__(**bind)
        self.l1 = Linear(d_in, d_hidden, True, id="l1", batch=B)
        self.l2 = Linear(d_hidden, d_out, False, id="l2", batch=B)
        self.loss = SquaredError(d_out, id="loss", batch=B)
        self.inputs = {"x": self.l1.inputs["x"], "target": self.loss.inputs["target"]}
        self.outputs = {"loss": self.loss.outputs["loss"]}

    def forward(self, x, target):
        return self.loss(self.l2(self.l1(x)), target=target)
