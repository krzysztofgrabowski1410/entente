"""Game engine: board model, orders, phases, and the adjudication core.

Constraints that define this package:
- stdlib only -- no third-party runtime dependencies;
- fully deterministic -- all randomness lives in agents or seeded engine-level
  variant hooks, never in adjudication;
- ruleset-parameterized -- classic Diplomacy is one instantiation of the
  generalized calculus (see entente.engine.rules), certified by DATC.
"""
