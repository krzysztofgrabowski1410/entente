"""Adjudication: two implementations of one spec (docs/adjudication.tex).

The staged fixpoint resolver (``resolver``, Sec. 6) is the default engine; the
stable-labeling enumerator (``csp``, Sec. 5) is the executable spec and permanent
differential oracle -- both remain fully supported and share Phase 0-1, the
Sec. 5.1 ``Eval`` and result assembly (``common``), so ``resolver == enumerator``
tests exactly the labeling search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from entente.engine.adjudicator import csp, resolver
from entente.engine.adjudicator.common import (
    AdjudicationError,
    Dislodgement,
    OrderResolution,
    PhaseResult,
    Resolution,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from entente.engine.map.graph import MapGraph
    from entente.engine.orders import Order
    from entente.engine.rules.protocol import Ruleset
    from entente.engine.state import State

EngineName = Literal["resolver", "enumerator"]

ENGINES = {
    "resolver": resolver.adjudicate,
    "enumerator": csp.adjudicate,
}
"""Engine name -> implementation; both adjudicate the same spec."""


def adjudicate(
    ruleset: Ruleset,
    state: State,
    sheets: Mapping[str, Sequence[Order]],
    *,
    graph: MapGraph | None = None,
    engine: EngineName = "resolver",
) -> PhaseResult:
    """Adjudicate one movement phase (docs/adjudication.tex Sec. 5/Sec. 6) -- pure function.

    ``engine`` selects the implementation: the staged fixpoint ``"resolver"``
    (default) or the stable-labeling ``"enumerator"`` (the executable spec;
    kept as the differential oracle). Both are deterministic in
    (ruleset, state, sheets) and observationally identical.
    """
    try:
        implementation = ENGINES[engine]
    except KeyError:
        raise ValueError(f"unknown adjudication engine {engine!r}") from None
    return implementation(ruleset, state, sheets, graph=graph)


__all__ = [
    "ENGINES",
    "AdjudicationError",
    "Dislodgement",
    "EngineName",
    "OrderResolution",
    "PhaseResult",
    "Resolution",
    "adjudicate",
]
