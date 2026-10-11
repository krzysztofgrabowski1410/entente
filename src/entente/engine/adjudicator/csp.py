"""The stable-labeling enumerator -- the executable form of docs/adjudication.tex Sec. 5.

``adjudicate`` implements Sec. 5/Sec. 6 literally: Phase 0-1 from ``common``, then per
SCC of the order-dependency graph (processed dependencies-first): propagate
the cheap forced facts, then run the shared Sec. 5.1 selection
(``common.resolve_residue``) over the component's undecided orders --
enumerate SUCCEEDS/FAILS assignments, keep exactly the stable ones
(L = Eval(L)), take a unique one, commit unanimity and re-enumerate on
several, and hand only genuine ambiguity to the ruleset's backup policy
(Sec. 5.2). Exponential enumeration is confined to SCCs, which are tiny in
practice -- that is the design: correctness first; the staged resolver
(``resolver``) is the fast implementation and this stays its differential
oracle.

Enumeration ranges over *relocating* orders only: Sec. 5.1 makes every stationary
status a derived predicate of the move labels (supports via cut/void, convoys
via fleet dislodgement and route use, holds via dislodgement), so the stable
labelings over all orders are in bijection with the stable move assignments.
The ``Eval`` itself (``common.Evaluator``), the residue selection
(``common.resolve_residue`` -- the *selection*, not just the backup policy,
is shared, so both engines settle residues at the identical Sec. 5.1 point)
and result assembly are shared with the resolver; this module owns only the
per-component drive and the forced-facts pre-pass.

Determinism: records are canonically ordered, adjacency lists sorted, Tarjan
emission fixed, residue selection applied to canonically sorted members -- no
wall-clock, no RNG, no set-iteration dependence anywhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from entente.engine.adjudicator.common import (
    Evaluator,
    PhaseResult,
    assemble,
    build_tables,
    condensation_order,
    dependency_graph,
    phase0,
    possible_chain_for,
    resolve_residue,
)
from entente.engine.map.loader import load_map
from entente.engine.orders import Move
from entente.engine.state import State, check_state

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from entente.engine.map.graph import MapGraph
    from entente.engine.orders import Order
    from entente.engine.rules.protocol import Ruleset


def _resolve_component(evaluator: Evaluator, members: Sequence[int]) -> None:
    """Resolve one SCC: cheap forced facts, then the shared Sec. 5.1 selection (Sec. 6).

    The backup policy only ever sees the genuine ambiguity.
    """
    undecided: list[int] = []
    for index in sorted(members):
        if not evaluator.is_move(index) or evaluator.status[index] is not None:
            continue
        if evaluator.is_convoyed(index) and not possible_chain_for(evaluator, index):
            evaluator.status[index] = False  # forced: no chain in any labeling
            continue
        destination = evaluator.entry_target(index)
        rivals = [r for r in evaluator.tables.into.get(destination, ()) if r != index]
        occupied = evaluator.tables.records_at.get(destination, ())
        bars = evaluator.tables.bars_on.get(index, ())
        if (
            isinstance(evaluator.records[index].effective, Move)
            and not evaluator.is_convoyed(index)
            and not rivals
            and not occupied
            and not bars
        ):
            evaluator.status[index] = True  # forced: uncontested entry into empty province
            continue
        undecided.append(index)
    resolve_residue(evaluator, undecided)


def adjudicate(
    ruleset: Ruleset,
    state: State,
    sheets: Mapping[str, Sequence[Order]],
    *,
    graph: MapGraph | None = None,
) -> PhaseResult:
    """Adjudicate one movement phase (docs/adjudication.tex Sec. 5/Sec. 6) -- pure function.

    ``sheets`` maps power -> *ordered* order list (Sec. 1); units without orders
    hold. The result is deterministic in (ruleset, state, sheets): same
    inputs, same ``PhaseResult``, byte for byte.
    """
    if graph is None:
        graph = load_map(state.map_name)
    check_state(graph, state)
    parsed = phase0(graph, ruleset, state, sheets)
    tables = build_tables(graph, ruleset, parsed.records)
    evaluator = Evaluator(graph, ruleset, parsed.records, tables)
    adjacency = dependency_graph(parsed.records)
    for component in condensation_order(adjacency):
        _resolve_component(evaluator, component)
    return assemble(graph, state, sheets, parsed, evaluator)
