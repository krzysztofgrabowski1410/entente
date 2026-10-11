"""Provide the behavioral-cloning training stack.

Everything here is part of the ``ml`` dependency group's surface: the config
loader (:mod:`entente.training.config`) is torch-free and loads in any
environment, but :mod:`entente.training.bc` and :mod:`entente.training.tracking`
import torch / wandb and are exercised only by ``ml``-marked tests. Nothing in
this package is imported by the engine or the default CLI path except the config
models, which the tournament loader shares so the project has one config
mechanism rather than two.
"""
