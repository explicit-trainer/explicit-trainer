"""The framework: a model of modules with declared tensors (abstract/module.py), a placement of every
tensor (abstract/placement.py), implementations of modules as steps (abstract/module_impl.py,
abstract/step.py), a plan that builds one training step as a static graph (abstract/training_plan.py),
a schedule that places the steps on streams (schedule.py), and the trainer that runs them
(trainer.py). docs/framework.md explains the design.
"""
