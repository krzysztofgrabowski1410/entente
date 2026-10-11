"""Convoy chain geometry: the one sea walk that every convoy question shares.

Three questions in the calculus reduce to "do these seas link these coasts?":
Phase 0 legality (may an unflagged army move become a convoy attempt at all?),
resolution-time viability (does a chain of *undislodged* fleets exist?), and
route use (did a fleet's convoy actually carry anything?). A fourth asks *where*
they link them: from which seas a move's ordered convoys can put the army ashore,
the set the Beleriand convoy toll is read off (Sec. 2). They differ only in which
seas are offered and in whether the first goal reached is answer enough, so they
share one walk -- :func:`sea_chain_exists` stops it at the first goal,
:func:`sea_chain_ends` runs it out -- and none of them owns it, which is why this
sits beside ``results`` as the package's other leaf, importable by Phase 0, the
tables, the evaluator and the residue selection alike
(docs/adjudication.tex Sec. 3.5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from entente.engine.map.types import Location

if TYPE_CHECKING:
    from entente.engine.map.graph import MapGraph


def _chain_ends(
    graph: MapGraph,
    seas: set[str],
    anchors: frozenset[str],
    goals: frozenset[str],
    *,
    first_only: bool,
) -> list[str]:
    """Walk ``seas`` by fleet adjacency from the ``anchors`` seas; return the ``goals`` seas met.

    With ``first_only`` the walk stops at the first goal it meets. That is all
    a yes-or-no question needs, and the yes-or-no question has to stay cheap:
    it is asked once per legality check, once per enumerated convoy, and once
    per convoyed move under every tentative labeling the enumerator tries. A
    plain loop rather than a generator for the same reason -- suspending and
    resuming a frame costs more than the one-fleet walk itself. The frontier
    and its expansion are sorted, so the walk is deterministic.
    """
    frontier = sorted(seas & anchors)
    seen = set(frontier)
    ends: list[str] = []
    while frontier:
        sea = frontier.pop()
        if sea in goals:
            ends.append(sea)
            if first_only:
                break
        for neighbor in sorted(seas - seen):
            if graph.fleet_reachable(Location(sea), Location(neighbor)):
                seen.add(neighbor)
                frontier.append(neighbor)
    return ends


def sea_chain_exists(
    graph: MapGraph,
    seas: set[str],
    anchors: frozenset[str],
    goals: frozenset[str],
) -> bool:
    """Whether ``seas`` holds a fleet-adjacency chain from an ``anchors`` sea to a ``goals`` sea.

    One shared walk serves the legality, viability and route-use checks of
    convoy geometry (Sec. 3.5); it stops at the first goal it reaches.
    """
    return bool(_chain_ends(graph, seas, anchors, goals, first_only=True))


def sea_chain_ends(
    graph: MapGraph,
    seas: set[str],
    anchors: frozenset[str],
    goals: frozenset[str],
) -> frozenset[str]:
    """Return every ``goals`` sea in which a chain through ``seas`` from an ``anchors`` sea ends.

    The same walk as :func:`sea_chain_exists`, run until nothing new is
    reached: non-empty exactly when that one is true. A goal sea reached
    through another goal sea counts as well -- a chain may pass one landing
    sea and end in the next.
    """
    return frozenset(_chain_ends(graph, seas, anchors, goals, first_only=False))


def potential_chain(
    graph: MapGraph,
    capable_seas: frozenset[str],
    origin: str,
    destination: str,
) -> bool:
    """Whether fleets present on the board could form a convoy chain from origin to destination.

    Orders are ignored: this is the Sec. 1 legality input. ``capable_seas``
    holds the sea provinces occupied by fleets able to carry the cargo
    (``Ruleset.convoy_capability`` -- any fleet under the standard rules;
    Beleriand needs Double Fleets for a Double Army, Sec. 3.5). The answer
    separates DATC 4.E.1's two buckets: a non-adjacent army move with no such
    chain is *illegal* (ignored -- the unit is treated as unordered,
    6.D.28-6.D.32), while with a chain it is a convoy attempt that merely fails
    (the unit counts as moving, 6.D.8).
    """
    return sea_chain_exists(
        graph, set(capable_seas), graph.sea_neighbors(origin), graph.sea_neighbors(destination)
    )
