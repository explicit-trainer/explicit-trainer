"""The placement of the example: data parallel over the mesh dimension "gpu".

Purpose and design. Every weight rests whole in FP32 on every device ("value", persistent) and has
a gradient that each device holds as its partial sum ("grad", for the step). Every port with a batch
axis is split along it over "gpu"; the scalar loss is held whole.

Public components
    DataParallel(model, devices)
        mesh = {"gpu": devices}; place(tensor_id) -> the copies by name.

Example
    placement = DataParallel(net, devices=2)
    placement.place("net.l1.w")["grad"].layout.partial    # ("gpu",)
"""
from explicit_trainer.framework.abstract.placement import Copy, Layout, Placement


class DataParallel(Placement):
    def __init__(self, model, devices: int):
        super().__init__(model)
        self.mesh = {"gpu": devices}

    def place(self, tensor_id: str) -> dict:
        module, name = self.model.resolve(tensor_id)
        if name in getattr(module, "state", {}):
            return {"value": Copy("value", Layout(), "persistent"),
                    "grad": Copy("grad", Layout(partial=("gpu",)), "step")}
        port = module.inputs[name] if name in module.inputs else module.outputs[name]
        split = {"batch": ("gpu",)} if "batch" in port.axes else {}
        return {"value": Copy("value", Layout(split), "backward")}
