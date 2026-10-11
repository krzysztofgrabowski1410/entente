"""entente -- a ruleset-generalized Diplomacy engine and AI research project.

Layers, lowest first:
  engine     -- board model + adjudication core, stdlib only, parameterized by ruleset
  variants   -- rulebook instantiations of the calculus (Beleriand)
  gametheory -- regret minimization / CFR educational modules with convergence tests
  agents     -- random / heuristic / behavioral-cloning / piKL-search agents
  nn, data, training, search, eval -- the ML stack (standard ruleset only)
  negotiation -- optional LLM negotiation prototype (isolated, cuttable)

Importing the package wires the variant registry: the engine never imports
variants, so registration lives here at the top level.
"""

import entente.variants.beleriand  # noqa: F401  - registers the ruleset (import side effect)

__version__ = "0.0.1"
