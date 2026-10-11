"""The generalization layer: everything rule-specific flows through a Ruleset.

Two concrete rulesets drive the interface (the "rule of two"): classic
Diplomacy and the Beleriand variant (Layer 1). No speculative axes.
"""

from entente.engine.rules.protocol import (
    AdmissionPolicy,
    BackupAction,
    CutExceptionScope,
    ResidueClass,
    Ruleset,
    StrengthRole,
)
from entente.engine.rules.standard import STANDARD, StandardRuleset

RULESETS: dict[str, Ruleset] = {STANDARD.name: STANDARD}
"""Ruleset name -> instance; variant packages register here on import
(``entente/__init__`` wires them -- the engine never imports variants)."""

__all__ = [
    "RULESETS",
    "STANDARD",
    "AdmissionPolicy",
    "BackupAction",
    "CutExceptionScope",
    "ResidueClass",
    "Ruleset",
    "StandardRuleset",
    "StrengthRole",
]
