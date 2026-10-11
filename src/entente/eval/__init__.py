"""Evaluation harness: run the games, score them, compare two agents honestly.

Tournaments run over :mod:`entente.sim.parallel`; games are scored by the
Diplomacy measures (Sum-of-Squares, survival, solo); and the statistics
(bootstrap confidence intervals, permutation tests) are what turn raw outcomes
into a defensible head-to-head comparison.

The standard ruleset is the only one supported, by design: seat rotation, the
agent maps and the scoring keys all name the standard seven powers.
"""
