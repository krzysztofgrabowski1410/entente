"""Game-theory primitives: regret-based learners with known-answer validation.

Every learner here is checked on matrix games and Kuhn poker, whose equilibria
are known in closed form, before any of it is trusted on Diplomacy. The one
downstream consumer is :mod:`entente.search.pikl`, which reuses
:mod:`entente.gametheory.regret`'s Hedge learner KL-anchored to a base policy.

Stdlib-only, like :mod:`entente.engine`, and with no engine dependency either:
these modules are pure game theory and know nothing about Diplomacy.
"""
