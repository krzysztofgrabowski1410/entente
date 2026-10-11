"""Phase 1: the order-dependency digraph and its SCC condensation.

The digraph is built over *orders, from order syntax* -- never from map
adjacency (docs/adjudication.tex Sec. 6, R7). It overapproximates every read
``Eval`` can make (Lemma L1), so SCC-independence implies true independence,
components can be resolved in dependency order, and cycles/paradoxes are
confined to SCCs. Tarjan emits SCCs dependencies-first and the whole
construction is deterministic (canonical record order, sorted adjacency).

:func:`entry_province` lives here because Phase 1 is its first consumer, but it
is a plain order accessor -- "which province does this order contest?" -- and
the tables, the evaluator and result assembly all reuse it.

This module changes only for graph-theoretic reasons (what may depend on
what), independent of any rule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from entente.engine.orders import Bar, Convoy, Dislodge, Move, Order, Support

if TYPE_CHECKING:
    from collections.abc import Sequence

    from entente.engine.adjudicator.common.results import OrderRecord


def entry_province(order: Order) -> str | None:
    """Return the province a relocating order contests entry at, or None for a stationary one.

    A move's destination; a Dislodge's victim Area (Sec. 4.3).
    """
    if isinstance(order, Move):
        return order.destination.province
    if isinstance(order, Dislodge):
        return order.victim.location.province
    return None


def dependency_graph(records: Sequence[OrderRecord]) -> tuple[tuple[int, ...], ...]:
    """Phase 1 (Sec. 6): the static order-dependency digraph, from order syntax.

    Edge ``i -> j`` means Eval may read j's status while evaluating i. The
    edge lists follow the Sec. 6 Phase-1 table, overapproximated syntactically
    (L1): a relocating order depends on rivals into its destination (moves
    and Dislodges alike), occupants of its destination, supports, convoys
    and Bars aimed there, the counter-move corridor, and relocations into
    its own origin (prevent-nulling), and the supports aimed at the home of
    each Dislodge into its destination (the defend of that Dislodge's victim
    decides whether the Dislodge lost its clash and so leaves the Area's
    admission); supports, convoys and bars depend on
    the relocating orders threatening their province (and, for a convoy, on
    its sibling convoys' provinces through those siblings), and on the
    supports aimed at the home of each Dislodge among those threats (whether
    that Dislodge lost its clash decides whether it takes part in a twin walk).
    """
    at_province: dict[str, list[int]] = {}
    into: dict[str, list[int]] = {}
    supports_toward: dict[str, list[int]] = {}
    convoys_toward: dict[str, list[int]] = {}
    convoys_by_cargo: dict[tuple[str, str], list[int]] = {}
    bars_at: dict[str, list[int]] = {}
    dislodge_homes: dict[str, set[str]] = {}
    for record in records:
        at_province.setdefault(record.unit.location.province, []).append(record.index)
        order = record.effective
        target = entry_province(order)
        if target is not None:
            into.setdefault(target, []).append(record.index)
            if isinstance(order, Dislodge):
                dislodge_homes.setdefault(target, set()).add(record.unit.location.province)
        elif isinstance(order, Bar):
            bars_at.setdefault(order.destination.province, []).append(record.index)
        elif isinstance(order, Support):
            subject = order.subject
            subject_target = entry_province(subject)
            if subject_target is None:
                subject_target = (
                    subject.destination.province
                    if isinstance(subject, Bar)
                    else subject.location.province
                )
            supports_toward.setdefault(subject_target, []).append(record.index)
        elif isinstance(order, Convoy):
            cargo = order.cargo
            key = (cargo.location.province, cargo.destination.province)
            convoys_toward.setdefault(cargo.destination.province, []).append(record.index)
            convoys_by_cargo.setdefault(key, []).append(record.index)

    edges: list[set[int]] = [set() for _ in records]
    for record in records:
        deps = edges[record.index]
        order = record.effective
        target = entry_province(order)
        if target is not None:
            origin = record.unit.location.province
            deps.update(into.get(target, ()))  # rivals (moves and dislodges)
            deps.update(at_province.get(target, ()))  # stayers / defenders / counter-move
            deps.update(supports_toward.get(target, ()))  # supports of rivals and defenders
            deps.update(convoys_toward.get(target, ()))  # convoys carrying rivals
            deps.update(bars_at.get(target, ()))  # Bars naming entrants
            deps.update(supports_toward.get(origin, ()))  # counter-move's supports
            deps.update(into.get(origin, ()))  # own dislodgement -> prevent-nulling
            for home in dislodge_homes.get(target, ()):
                deps.update(supports_toward.get(home, ()))  # a rival Dislodge's clash
            if isinstance(order, Move):
                deps.update(convoys_by_cargo.get((origin, target), ()))  # own chain
        else:
            province = record.unit.location.province
            deps.update(into.get(province, ()))  # threats to this unit
            deps.update(supports_toward.get(province, ()))  # their supports
            deps.update(convoys_toward.get(province, ()))  # convoyed threats' chains
            for home in dislodge_homes.get(province, ()):
                deps.update(supports_toward.get(home, ()))  # a threatening Dislodge's clash
            if isinstance(order, Convoy):
                cargo = order.cargo
                key = (cargo.location.province, cargo.destination.province)
                deps.update(convoys_by_cargo.get(key, ()))  # sibling fleets
        deps.discard(record.index)
    return tuple(tuple(sorted(dep_set)) for dep_set in edges)


def condensation_order(adjacency: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Return the SCCs of the dependency digraph, dependencies first, deterministically.

    Iterative Tarjan over canonical indices with sorted adjacency: an SCC is
    emitted only after every SCC it depends on, so components can be resolved
    in emission order (Sec. 6 Phase 1). Members are returned sorted.
    """
    n = len(adjacency)
    index_of = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack: list[int] = []
    components: list[tuple[int, ...]] = []
    counter = 0
    for root in range(n):
        if index_of[root] != -1:
            continue
        work: list[tuple[int, int]] = [(root, 0)]
        while work:
            node, child_pos = work[-1]
            if child_pos == 0:
                index_of[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack[node] = True
            advanced = False
            for position in range(child_pos, len(adjacency[node])):
                neighbor = adjacency[node][position]
                if index_of[neighbor] == -1:
                    work[-1] = (node, position + 1)
                    work.append((neighbor, 0))
                    advanced = True
                    break
                if on_stack[neighbor]:
                    low[node] = min(low[node], index_of[neighbor])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index_of[node]:
                component: list[int] = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == node:
                        break
                components.append(tuple(sorted(component)))
    return tuple(components)
