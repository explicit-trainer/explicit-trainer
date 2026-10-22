"""A complete small example of explicit-trainer: a two-layer perceptron with a squared-error loss,
data parallel over a mesh {"gpu": n}, that trains and checks on CPU with gloo (torchrun, n processes).

Modules
    model.py        the model: Linear and SquaredError leaves, the Net composite
    placement.py    DataParallel: FP32 weights replicated, gradients as partial sums, the batch split
    impl.py         LinearImpl and SquaredErrorImpl: forward and backward as steps
    plan.py         TinyPlan: one training step (micro-batches, all-reduce of gradients, SGD)
    schedule.py     generate(): a schedule in program order
    run.py          trains a few steps and checks the step against the references
"""
