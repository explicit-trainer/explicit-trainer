"""The abstractions every model, placement, implementation and plan is written against.

Modules
    module.py         the model: modules, ports, states, references, coarse leaves
    placement.py      where every tensor lives: copies and their layouts
    step.py           the vocabulary of work: steps, values, copy references, graphs
    module_impl.py    how a module is computed, as steps
    training_plan.py  one training step as one graph of steps
"""
