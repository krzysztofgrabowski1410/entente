"""A heuristic ("greedy") Diplomacy agent -- the deterministic evaluation baseline.

It is the reference every other policy is measured against, alongside
:class:`~entente.agents.random_agent.RandomAgent`: an agent that cannot beat a
one-round heuristic has shown nothing yet.

Own scoring design: every province gets a static value from supply-center
ownership (enemy-held centers outrank neutral ones; a province already ours,
or with no center at all, contributes nothing on its own), then that value
is spread one hop to its neighbours so ground *near* a center matters too,
not only centers themselves. Each unit's candidate action is scored by the
diffused value of where it ends up, net of a contest-risk term (how many
enemy units sit adjacent to the destination) and, for holding an owned,
under-defended, threatened center, a defense bonus. Every unit picks its
best action independently first (this pass never considers a via-convoy
move, since a lone army cannot arrange its own convoy -- see
:meth:`GreedyAgent._best_independent_action`); a second pass then looks for
units whose independent action scored poorly and redirects them to
*support* a higher-scoring neighbouring friendly action instead, when that
support is legal and the target looks contested (otherwise the support
would cut nothing). That second pass is this agent's whole answer to
"support coordination" -- it is local by design (one round, no search
over joint plans) so that this stays a *baseline*, not a search agent.

Public prose descriptions of "DumbBot"-class heuristic Diplomacy bots
(never code) were read for the general shape a simple heuristic bot takes --
value provinces, spread value across the map, move toward the best
reachable value -- e.g. the DAIDE community's algorithm summary
(http://www.daide.org.uk/s0003.html) and academic surveys that describe it
as a baseline (e.g. X. Croes, *Tree Search Methods for Diplomacy Agents*,
MSc thesis, Leiden University, 2016). The value function, diffusion rule,
risk/defense terms and the two-pass support-coordination procedure below
are this project's own design, not a transcription of any of those: the
heuristic is original and no DumbBot source was consulted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from entente.engine.orders import Build, Disband, Hold, Move, Support, Waive
from entente.engine.phase import PhaseKind

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from entente.engine.game import Game
    from entente.engine.map.graph import MapGraph
    from entente.engine.orders import AnyOrder, SupportableOrder
    from entente.engine.state import State

NEUTRAL_SC_VALUE = 1.0
"""Value of an unowned supply center -- worth taking, but denying an enemy
one is worth more (below)."""
ENEMY_SC_VALUE = 1.4
"""Value of an enemy-owned supply center: taking it both grows this power
and shrinks the opponent's, so it outranks an equally-reachable neutral."""
DIFFUSION_WEIGHT = 0.35
"""How much of a neighbor's value spreads to a province in
:func:`_diffused_values` -- high enough to draw units toward contested
regions before a center is directly reachable, low enough that a center's
own value still dominates its neighbours'."""
RISK_WEIGHT = 0.4
"""Score subtracted per enemy unit adjacent to a candidate move's
destination -- a stand-in for "this attack is likely contested/supported"
without predicting the opponent's actual orders."""
DEFENSE_WEIGHT = 0.6
"""Bonus (per adjacent enemy unit) added to holding an owned, single- or
un-defended, threatened center -- enough to usually outweigh a modestly
better attack elsewhere, not so much that defense always wins."""
DISBAND_SCORE = -1.0
"""Fixed score for retreat-phase Disband -- below any real destination's
score (>= 0), so a dislodged unit only ever disbands when no retreat is
legal at all (`legal_orders` always lists at least Disband, so this never
crashes; :meth:`GreedyAgent._retreat_orders` just never prefers it when a
real square is open)."""


def _touching(graph: MapGraph, province: str) -> frozenset[str]:
    """Return every province adjacent to ``province`` for any unit kind.

    This is the neighbourhood :func:`_diffused_values` and the risk and
    defense terms spread over. Reading armies and fleets as one union is a
    simplification by design: it is enough to bias play toward contested
    regions without a per-unit-kind reachability query, for a heuristic that
    never plans convoys anyway.
    """
    return graph.army_adjacency.get(province, frozenset()) | graph.sea_neighbors(province)


def _base_value(graph: MapGraph, ownership: Mapping[str, str], power: str, province: str) -> float:
    """Return ``province``'s standing worth to ``power``, before any diffusion."""
    prov = graph.province(province)
    if not prov.is_supply_center:
        return 0.0
    owner = ownership.get(province)
    if owner == power:
        return 0.0
    return NEUTRAL_SC_VALUE if owner is None else ENEMY_SC_VALUE


def _diffused_values(graph: MapGraph, ownership: Mapping[str, str], power: str) -> dict[str, float]:
    """Return every province's value: its own worth plus a share of its neighbours'.

    The share is ``DIFFUSION_WEIGHT`` times the average :func:`_base_value` of
    the adjacent provinces.
    """
    base = {
        province: _base_value(graph, ownership, power, province) for province in graph.provinces
    }
    diffused: dict[str, float] = {}
    for province in graph.provinces:
        neighbors = _touching(graph, province)
        spread = sum(base[n] for n in neighbors) / len(neighbors) if neighbors else 0.0
        diffused[province] = base[province] + DIFFUSION_WEIGHT * spread
    return diffused


def _enemy_units_adjacent(graph: MapGraph, state: State, power: str, province: str) -> int:
    neighbors = _touching(graph, province)
    return sum(
        1 for unit in state.units if unit.power != power and unit.location.province in neighbors
    )


def _defense_bonus(
    graph: MapGraph, ownership: Mapping[str, str], state: State, power: str, province: str
) -> float:
    """Return the bonus for holding an owned, threatened, thinly held centre.

    It applies only where ``power`` owns the province, at least one enemy unit
    stands adjacent, and no more than one friendly unit is there to answer: a
    province already held in force needs no extra defender.
    """
    if ownership.get(province) != power:
        return 0.0
    threats = _enemy_units_adjacent(graph, state, power, province)
    if threats == 0:
        return 0.0
    defenders = sum(1 for unit in state.occupants(province) if unit.power == power)
    return DEFENSE_WEIGHT * threats if defenders <= 1 else 0.0


def _group_by_unit(orders: Iterable[AnyOrder]) -> dict[str, list[AnyOrder]]:
    """Group a movement- or retreat-phase ``legal_orders`` sheet by ordering unit.

    Exactly one entry per group is picked. Neither call site passes an
    adjustment sheet (``_adjustment_orders`` reads ``legal`` directly), but the
    parameter is typed as the full ``AnyOrder`` union, so the ``Waive`` guard
    stays: narrowing ``orders`` to a movement/retreat-only type would mean
    threading that narrower type through both call sites for no behavioural
    difference.
    """
    by_unit: dict[str, list[AnyOrder]] = {}
    for order in orders:
        if isinstance(order, Waive):  # pragma: no cover - unreachable via either call site
            continue
        by_unit.setdefault(str(order.location), []).append(order)
    return by_unit


def _matches_chosen(chosen_order: AnyOrder | None, subject: SupportableOrder) -> bool:
    """Report whether ``subject`` names the action ``chosen_order`` is taking.

    A Move subject's ``via_convoy`` flag is ignored on both sides: support
    notation always normalizes it to False whatever route the move really
    takes, so comparing it would reject a legal support of a convoyed move.
    """
    if chosen_order is None:
        return False
    if isinstance(subject, Move) and isinstance(chosen_order, Move):
        return chosen_order.destination == subject.destination
    return chosen_order == subject


class GreedyAgent:
    """A heuristic agent with no randomness: one position always gives one sheet.

    Unlike :class:`~entente.agents.random_agent.RandomAgent`, two instances
    replay identically with no seed dependence at all. ``seed`` is accepted
    only so that ``Callable[[int], Agent]`` factories, such as
    :mod:`entente.agents.registry`, can build every agent the same way.
    """

    def __init__(self, seed: int) -> None:
        del seed  # deterministic; kept for a uniform seed -> Agent factory signature

    def orders(self, game: Game, power: str) -> Sequence[AnyOrder]:
        """Return ``power``'s sheet for ``game``'s current phase."""
        legal = game.legal_orders(power)
        if not legal:
            return []
        kind = game.phase_kind
        if kind is PhaseKind.MOVEMENT:
            return self._movement_orders(game, power, legal)
        if kind is PhaseKind.RETREAT:
            return self._retreat_orders(game, power, legal)
        return self._adjustment_orders(game, power, legal)

    # -- movement -------------------------------------------------------------

    def _movement_orders(self, game: Game, power: str, legal: Sequence[AnyOrder]) -> list[AnyOrder]:
        values = _diffused_values(game.graph, game.ownership, power)
        by_unit = _group_by_unit(legal)

        chosen: dict[str, AnyOrder] = {}
        scores: dict[str, float] = {}
        for location, options in by_unit.items():
            order, score = self._best_independent_action(game, power, values, options)
            chosen[location] = order
            scores[location] = score

        self._coordinate_supports(game, power, values, by_unit, chosen, scores)
        return list(chosen.values())

    def _best_independent_action(
        self, game: Game, power: str, values: Mapping[str, float], options: Sequence[AnyOrder]
    ) -> tuple[AnyOrder, float]:
        """Return the best-scoring Hold or Move among ``options`` for one unit.

        Support candidates are left to :meth:`_coordinate_supports`; Convoy and
        any Beleriand Dislodge or Bar candidates are ignored, because this
        agent plans no contest of that kind and arranges no convoys.
        """
        graph = game.graph
        state = game.state
        best_order: AnyOrder | None = None
        best_score = float("-inf")
        for order in options:
            score: float
            if isinstance(order, Hold):
                province = order.location.province
                score = values[province] + _defense_bonus(
                    graph, game.ownership, state, power, province
                )
            elif isinstance(order, Move) and not order.via_convoy:
                destination = order.destination.province
                score = values[destination] - RISK_WEIGHT * _enemy_units_adjacent(
                    graph, state, power, destination
                )
            else:
                continue
            if score > best_score:
                best_order, best_score = order, score
        if best_order is None:  # pragma: no cover - every unit has a Hold option
            raise ValueError("no Hold/Move candidate among legal orders for a unit")
        return best_order, best_score

    def _coordinate_supports(
        self,
        game: Game,
        power: str,
        values: Mapping[str, float],
        by_unit: Mapping[str, list[AnyOrder]],
        chosen: dict[str, AnyOrder],
        scores: Mapping[str, float],
    ) -> None:
        """Redirect units into supporting a friendly neighbour's already chosen action.

        A unit switches only when that support is legal, the target looks
        contested (an enemy unit stands adjacent to it -- an uncontested target
        needs no support, so redirecting would only waste the turn), and the
        support scores higher than the unit's own independent action did. A
        Move's score has already paid the contest-risk penalty for attacking
        that square alone, while a support of the exact same square is what
        wins the fight, so the support is compared against the raw target value
        with no such penalty of its own. Every unit is checked, not only ones
        whose own action scored poorly: a unit with a solid attack elsewhere is
        never redirected, since its own score already exceeds any lower-value
        target it could support, but one that could only match, not beat, a
        teammate's contested attack should reinforce it instead. Mutates
        ``chosen`` in place.
        """
        graph = game.graph
        state = game.state
        for location, options in by_unit.items():
            best_support: Support | None = None
            best_target_score = float("-inf")
            for order in options:
                if not isinstance(order, Support):
                    continue
                if not _matches_chosen(chosen.get(str(order.subject.location)), order.subject):
                    continue  # that unit isn't actually doing this, this turn
                target_province = (
                    order.subject.destination.province
                    if isinstance(order.subject, Move)
                    else order.subject.location.province
                )
                if _enemy_units_adjacent(graph, state, power, target_province) == 0:
                    continue
                target_score = values[target_province]
                if target_score > best_target_score:
                    best_support, best_target_score = order, target_score
            if best_support is not None and best_target_score > scores[location]:
                chosen[location] = best_support

    # -- retreats ---------------------------------------------------------------

    def _retreat_orders(self, game: Game, power: str, legal: Sequence[AnyOrder]) -> list[AnyOrder]:
        values = _diffused_values(game.graph, game.ownership, power)
        by_unit = _group_by_unit(legal)
        return [
            max(options, key=lambda order: self._retreat_score(order, values))
            for _, options in sorted(by_unit.items())
        ]

    def _retreat_score(self, order: AnyOrder, values: Mapping[str, float]) -> float:
        # Retreat destinations are always at least as good as disbanding
        # (DISBAND_SCORE < 0 <= any real destination's value); no risk term
        # here -- a retreat is fleeing to safety, not attacking.
        if isinstance(order, Disband):
            return DISBAND_SCORE
        assert isinstance(order, Move)
        return values[order.destination.province]

    # -- adjustments --------------------------------------------------------------

    def _adjustment_orders(
        self, game: Game, power: str, legal: Sequence[AnyOrder]
    ) -> list[AnyOrder]:
        quota = game.build_quota(power)
        values = _diffused_values(game.graph, game.ownership, power)
        if quota > 0:
            return self._build_orders(legal, values, quota)
        if quota < 0:
            return self._removal_orders(legal, values, -quota)
        return []  # pragma: no cover - legal_orders is empty whenever quota == 0

    def _build_orders(
        self, legal: Sequence[AnyOrder], values: Mapping[str, float], quota: int
    ) -> list[AnyOrder]:
        """Fill the quota from the highest-valued build sites, waiving any remainder."""
        builds = [order for order in legal if isinstance(order, Build)]
        ranked = sorted(builds, key=lambda b: values[b.location.province], reverse=True)
        chosen: list[AnyOrder] = []
        used_provinces: set[str] = set()
        for build in ranked:
            if len(chosen) == quota:
                break
            if build.location.province in used_provinces:
                continue  # one unit per home center per adjustment
            chosen.append(build)
            used_provinces.add(build.location.province)
        chosen.extend(Waive() for _ in range(quota - len(chosen)))
        return chosen

    def _removal_orders(
        self, legal: Sequence[AnyOrder], values: Mapping[str, float], count: int
    ) -> list[AnyOrder]:
        """Disband the ``count`` units standing in the lowest-valued provinces.

        Giving up the least useful positions first, and keeping the rest, is
        this agent's whole answer to a forced removal.
        """
        disbands = [order for order in legal if isinstance(order, Disband)]
        ranked = sorted(disbands, key=lambda d: values[d.location.province])
        return list(ranked[:count])
