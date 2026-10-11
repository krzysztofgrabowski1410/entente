"""Engine-only game rollouts, in two runners.

Which one to use is decided by what is driving the games:

- :mod:`entente.sim.parallel` -- a :class:`~concurrent.futures.ProcessPoolExecutor`
  runner for CPU-bound agents (random, greedy, search). Windows-spawn-safe.
- :mod:`entente.sim.vectorized` -- a single-process lockstep stepper for
  NN-in-the-loop rollouts, batching every pending decision into one policy
  callback per tick so a GPU policy runs one forward pass per tick instead of
  one per game. No torch dependency here.
"""

from entente.sim.parallel import GameOutcome, SimJob, run_parallel, run_sequential
from entente.sim.vectorized import DecisionRequest, LockstepRunner, PolicyCallback

__all__ = [
    "DecisionRequest",
    "GameOutcome",
    "LockstepRunner",
    "PolicyCallback",
    "SimJob",
    "run_parallel",
    "run_sequential",
]
