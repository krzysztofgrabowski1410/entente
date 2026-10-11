"""Batched 1-ply joint-profile evaluation (docs/pikl.tex Sec. 6).

The utility of a joint candidate profile is the value function's score of the
successor position -- Jacob et al. 2022 Sec. 4.3.1's procedure at rollout length 0
(following Gray et al. 2021): step the engine once, score the resulting state,
no deeper rollout. :func:`profile_utility_fn` adapts that to the
:data:`entente.search.pikl.ProfileUtilityFn` oracle interface: every profile
in a batch becomes an independent branched game, all successors are scored in
**one** ``value_fn`` call (the GPU batching seam), and per-power utilities are
read off the returned mappings.

``value_fn`` is any callable -- production passes the BC value head via
:mod:`entente.agents.pikl_agent`; tests pass exact functions (supply-center
shares, terminal Sum-of-Squares), which is what lets the validation suite run
without torch. The contract: it receives the branched games (some possibly
already over -- a 1-ply step can end the game) and must return one
power-to-score mapping per game, covering at least the subgame's powers.

Branching reconstructs a fresh :class:`~entente.engine.game.Game` from the
read surface (``ruleset``/``graph``/``state``/``ownership``/``phase``) -- valid
exactly for **movement** phases, where adjudication reads no prior phase
result; the original game object is never mutated (its ``state`` is frozen and
``ownership`` is copied on read). Such branches are ephemeral search nodes:
they carry no replayable history and must never be serialized
(:mod:`entente.engine.serialization` refuses them by design).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from entente.engine.game import Game
from entente.engine.phase import Phase, PhaseKind

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from entente.engine.orders import AnyOrder
    from entente.search.pikl import ProfileUtilityFn

ValueFn: TypeAlias = "Callable[[Sequence[Game]], Sequence[Mapping[str, float]]]"
"""Batch state evaluator: one score-per-power mapping per game, same order."""


def branch_game(game: Game, sheets: Mapping[str, Sequence[AnyOrder]]) -> Game:
    """Apply a joint order profile to a copy of ``game`` and return the successor.

    The successor is advanced past any decision-free phases by the phase
    machine. Only a movement phase can be branched, because that is the only
    phase whose adjudication reads no prior phase result; anything else, and a
    finished game, raises ``ValueError``. Powers absent from ``sheets``, or
    mapped to an empty sheet, get the engine's phase defaults -- units hold.
    """
    if game.is_over:
        raise ValueError("cannot branch a finished game")
    if game.phase_kind is not PhaseKind.MOVEMENT:
        raise ValueError(f"1-ply branching is defined for movement phases, not {game.phase!r}")
    branch = Game(game.ruleset, game.graph, game.state, game.ownership, Phase.parse(game.phase))
    for power, sheet in sheets.items():
        if sheet:
            branch.submit(power, list(sheet))
    branch.adjudicate()
    return branch


def profile_utility_fn(
    game: Game,
    powers: Sequence[str],
    candidates: Sequence[Sequence[tuple[AnyOrder, ...]]],
    value_fn: ValueFn,
) -> ProfileUtilityFn:
    """Build the piKL utility oracle for one movement position.

    ``candidates[i]`` are ``powers[i]``'s candidate order-sets; a profile
    ``(c_0, ..., c_{n-1})`` selects one per power. The returned callable
    branches every profile in the batch, scores all successors with a single
    ``value_fn`` call, and returns utilities in ``powers`` order -- the
    player order :func:`entente.search.pikl.run_pikl` expects.
    """
    if len(powers) != len(candidates):
        raise ValueError("powers and candidates must be parallel")
    if len(set(powers)) != len(powers):
        raise ValueError("powers must be distinct")

    def evaluate(profiles: Sequence[tuple[int, ...]]) -> list[tuple[float, ...]]:
        branches = []
        for profile in profiles:
            if len(profile) != len(powers):
                raise ValueError(f"profile {profile!r} does not index {len(powers)} powers")
            sheets = {power: candidates[i][profile[i]] for i, power in enumerate(powers)}
            branches.append(branch_game(game, sheets))
        scores = value_fn(branches)
        if len(scores) != len(branches):
            raise ValueError(f"value_fn returned {len(scores)} scores for {len(branches)} games")
        return [tuple(float(score[power]) for power in powers) for score in scores]

    return evaluate


def supply_share_value(games: Sequence[Game]) -> list[dict[str, float]]:
    """Score each game by every power's share of the owned supply centers.

    A board where nobody owns a centre scores uniformly, which only happens
    before play begins. This torch-free :data:`ValueFn` is the exact-oracle
    stand-in the wiring tests use; production passes the BC value head instead.
    """
    results: list[dict[str, float]] = []
    for game in games:
        powers = game.powers
        counts = {power: float(game.supply_count(power)) for power in powers}
        total = sum(counts.values())
        if total <= 0.0:
            results.append({power: 1.0 / len(powers) for power in powers})
        else:
            results.append({power: count / total for power, count in counts.items()})
    return results


__all__ = ["ValueFn", "branch_game", "profile_utility_fn", "supply_share_value"]
